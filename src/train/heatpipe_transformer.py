import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
import sys, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import get_laplacian, degree, add_self_loops

# import path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from filepath import ABSOLUTE_PATH

sys.path.append(ABSOLUTE_PATH)
from src.train.train import Trainer
from src.utils.utils import create_res, set_seed, get_time, save_config_from_args, get_parameter_net, find_max_min
import time


def load_data(path, batchsize, tag, device, m=8):
    x = torch.tensor(np.load(path + "/data/heatpipe/x_element_as_one_node.npy")).to(device).float()
    y = torch.tensor(np.load(path + "/data/heatpipe/y_element_as_one_node.npy")).to(device).float()

    # ? Now all data share one graph
    edge_index = torch.tensor(
        np.load(path + "/data/heatpipe/adj_element_as_one_node.npy")[0].transpose(1, 0), dtype=torch.long
    ).to(device)
    edge_index, edge_weight = get_laplacian(edge_index)
    adjacency_matrix = torch.sparse_coo_tensor(edge_index, edge_weight)
    deg = torch.bincount(edge_index[0], minlength=16)
    deg_matrix = torch.diag(deg)
    laplacian_matrix = deg_matrix - adjacency_matrix

    eigvals, eigvecs = torch.linalg.eigh(laplacian_matrix)  # low to high
    top_eigvals = eigvals[-m:]  # top m
    top_eigvecs = eigvecs[:, -m:]
    eigvals_ep = top_eigvals.unsqueeze(0).repeat(top_eigvecs.shape[0], 1)
    PE = torch.cat((eigvals_ep.unsqueeze(1), top_eigvecs.unsqueeze(1)), dim=1)
    PE_all = PE.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

    train_dataset = TensorDataset(x[:tag], PE_all[:tag], y[:tag])

    test_dataset = TensorDataset(x[tag:], PE_all[tag:], y[tag:])

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)
    return train_loader, test_loader


def renormalize(x):
    if isinstance(x, torch.Tensor):
        x_n = x.clone()
    else:
        x_n = x.copy()
    bound = [[948, 1500], [0.00268, 0.00685], [0.00268, 0.00685]]
    # bound = [[948, 1500], [-4e-4, 1.2e-3], [-4e-4, 1.2e-3]]
    for i in range(3):
        x_n[..., i] = (x_n[..., i] + 1) / 2 * (bound[i][1] - bound[i][0]) + bound[i][0]
    return x_n


def get_loss(model, batch, loss_fn):
    x, pe, y = batch
    outputs_p = model(x, pe)
    loss = loss_fn(y, outputs_p)
    return loss


class SAN(nn.Module):
    """
    @article{kreuzer2021rethinking,
      title={Rethinking graph transformers with spectral attention},
      author={Kreuzer, Devin and Beaini, Dominique and Hamilton, Will and L{\'e}tourneau, Vincent and Tossou, Prudencio},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      pages={21618--21629},
      year={2021}
    }
    """

    def __init__(
        self,
        input_dim=4,
        hidden_dim=2412,
        output_dim=2412,
        k=32,
        m=8,
        num_heads_PE=4,
        dim_feedforward_PE=256,
        dropout_PE=0.1,
        num_layers_PE=2,
        num_heads_NF=8,
        dim_feedforward_NF=1024,
        dropout_NF=0.1,
        num_layers_NF=2,
    ):
        super().__init__()
        """
            k: embedding size
            m: top m eig value
        """
        self.k = k

        self.PE_linear = nn.Linear(2, k)
        self.NF_linear = nn.Linear(input_dim, hidden_dim - k)
        self.PE_transformer_layer = nn.TransformerEncoderLayer(
            d_model=m, nhead=num_heads_PE, dim_feedforward=dim_feedforward_PE, dropout=dropout_PE
        )
        self.PE_transformer = nn.TransformerEncoder(self.PE_transformer_layer, num_layers=num_layers_PE)
        self.NF_transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads_NF, dim_feedforward=dim_feedforward_NF, dropout=dropout_NF
        )
        self.NF_transformer = nn.TransformerEncoder(self.NF_transformer_layer, num_layers=num_layers_NF)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, PE):
        """_summary_

        Args:
            x (_type_): node feature, b, n, n_f
            PE (_type_): eigvals_vec, b,n,2,m
        """
        b, n, _, m = PE.shape
        LPE = PE.permute(0, 1, 3, 2)  # B,N,m,2
        LPE = self.PE_linear(LPE).permute(0, 1, 3, 2)  # B,N,k,m
        LPE = LPE.reshape(b * n, self.k, m)  # B*N,k,m
        LPE = self.PE_transformer(LPE).reshape(b, n, self.k, m)  # B,n,k,m
        LPE = torch.sum(LPE, dim=-1)  # B,N,k

        NF = self.NF_linear(x)  # B,N,nf-k

        NF_PE = torch.concat((NF, LPE), dim=-1)

        output = self.NF_transformer(NF_PE)
        return self.out(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train simple distribution model")
    parser.add_argument("--exp_id", default="heatpipe", type=str, help="experiment folder id")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--batchsize", default=128, type=int, help="size of dataset")
    parser.add_argument("--epoches", default=100000, type=int, help="training epoch")
    parser.add_argument("--checkpoint", default=2000, type=int, help="save and sample period")
    parser.add_argument("--overall_results_path", default="results", type=str, help="where to create result folder")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--in_dim", default=4, type=int, help="in dim")
    parser.add_argument("--hidden_dim", default=512, type=int, help="hidden dim")
    parser.add_argument("--out_dim", default=2412, type=int, help="output dim")

    parser.add_argument("--k", default=32, type=int, help="embedding dim")
    parser.add_argument("--m", default=8, type=int, help="top m eig value")

    parser.add_argument("--dim_feedforward_PE", default=64, type=int, help="dim_feedforward")
    parser.add_argument("--num_layers_PE", default=1, type=int, help="layers")
    parser.add_argument("--num_head_PE", default=4, type=int, help="num head")
    parser.add_argument("--dropout_PE", default=0.1, type=float, help="dropout")
    parser.add_argument("--dim_feedforward_NF", default=1024, type=int, help="dim_feedforward")
    parser.add_argument("--num_layers_NF", default=2, type=int, help="layers")
    parser.add_argument("--num_head_NF", default=8, type=int, help="num head")
    parser.add_argument("--dropout_NF", default=0.1, type=float, help="dropout")

    parser.add_argument("--num_node", default=804, type=int, help="num_node")
    parser.add_argument("--gradient_accumulate_every", default=2, type=int, help="gradient_accumulate_every")
    parser.add_argument("--gap", default=800, type=int, help="dataset size for train")
    parser.add_argument("--model_type", default="transformer", type=str, help="gnn or transformer")
    parser.add_argument("--paradigm", default="surrogate", type=str, help="diffusion or surrogate")
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_path = create_res(path=os.path.join(ABSOLUTE_PATH, args.overall_results_path), folder_name=args.exp_id)
    model_type = args.model_type
    paradigm = args.paradigm
    results_folder = os.path.join(
        results_path,
        paradigm,
        model_type + "_" + str(args.hidden_dim) + "_" + str(args.dim_feedforward_NF) + "_" + str(args.num_layers_NF),
    )
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    trainLoader, testLoader = load_data(
        path=ABSOLUTE_PATH, batchsize=args.batchsize, tag=args.gap, device=device, m=args.m
    )

    model = SAN(
        input_dim=args.in_dim,
        output_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        k=args.k,
        m=args.m,
        num_heads_PE=args.num_head_PE,
        dim_feedforward_PE=args.dim_feedforward_PE,
        dropout_PE=args.dropout_PE,
        num_layers_PE=args.num_layers_PE,
        num_heads_NF=args.num_head_NF,
        dim_feedforward_NF=args.dim_feedforward_NF,
        dropout_NF=args.dropout_NF,
        num_layers_NF=args.num_layers_NF,
    )
    get_parameter_net(model)
    train = Trainer(
        model=model,
        data_train=trainLoader,
        data_val=testLoader,
        train_function=get_loss,
        val_function=get_loss,
        train_lr=args.lr,
        gradient_accumulate_every=args.gradient_accumulate_every,
        train_num_steps=args.epoches,
        train_batch_size=args.batchsize,
        save_every=args.checkpoint,
        results_folder=results_folder,
    )
    train.train()
