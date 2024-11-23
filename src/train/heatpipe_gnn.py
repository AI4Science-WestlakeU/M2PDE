from torch_geometric.nn import GIN
import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
import sys, os

from torch_geometric.data import Data as Data_g
from torch_geometric.data import DataLoader as DataLoader_G
from torch.utils.data import DataLoader, TensorDataset


# import path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from filepath import ABSOLUTE_PATH

sys.path.append(ABSOLUTE_PATH)
from src.train.train import Trainer
from src.utils.utils import create_res, set_seed, get_time, save_config_from_args, get_parameter_net, find_max_min
import time


def get_loss(model: GIN, batch, loss_fn):
    outputs_p = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
    loss = loss_fn(batch.y, outputs_p)
    return loss


def load_data(path, batchsize, tag, device):
    x = torch.tensor(np.load(path + "/data/heatpipe/x_element_as_one_node.npy")).to(device).float()
    y = torch.tensor(np.load(path + "/data/heatpipe/y_element_as_one_node.npy")).to(device).float()
    edge_index = np.load(path + "/data/heatpipe/adj_element_as_one_node.npy").transpose(0, 2, 1)
    # print(np.max(edge_index), np.min(edge_index))
    edge_index = torch.tensor(edge_index).to(device).long()

    train_dataset = []
    for i in range(tag):
        data = Data_g(x=x[i], y=y[i], edge_index=edge_index[i])
        train_dataset.append(data)
    train_loader = DataLoader_G(train_dataset, batch_size=batchsize, shuffle=True)
    test_dataset = []
    for i in range(tag, x.shape[0]):
        data = Data_g(x=x[i], y=y[i], edge_index=edge_index[i])
        test_dataset.append(data)
    test_loader = DataLoader_G(test_dataset, batch_size=batchsize * 10, shuffle=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train simple distribution model")
    parser.add_argument("--exp_id", default="heatpipe", type=str, help="experiment folder id")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--batchsize", default=64, type=int, help="size of dataset")
    parser.add_argument("--epoches", default=20000, type=int, help="training epoch")
    parser.add_argument("--checkpoint", default=2000, type=int, help="save and sample period")
    parser.add_argument("--overall_results_path", default="results", type=str, help="where to create result folder")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--in_dim", default=4, type=int, help="in dim")
    parser.add_argument("--out_dim", default=2412, type=int, help="output dim")
    parser.add_argument("--hidden_dim", default=3600, type=int, help="hidden dim")
    parser.add_argument("--n_layer", default=4, type=int, help="layers")
    parser.add_argument("--slice_num", default=16, type=int, help="transolver slice_num")
    parser.add_argument("--num_node", default=804, type=int, help="num_node")
    parser.add_argument("--gradient_accumulate_every", default=2, type=int, help="gradient_accumulate_every")
    parser.add_argument("--gap", default=800, type=int, help="dataset size for train")
    parser.add_argument("--model_type", default="GIN_4_3600", type=str, help="gnn or transformer")
    parser.add_argument("--paradigm", default="surrogate", type=str, help="diffusion or surrogate")
    args = parser.parse_args()
    set_seed(args.seed)
    results_path = create_res(path=os.path.join(ABSOLUTE_PATH, args.overall_results_path), folder_name=args.exp_id)
    model_type = args.model_type
    paradigm = args.paradigm
    results_folder = os.path.join(results_path, paradigm, model_type)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    model = GIN(
        in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim, num_layers=args.n_layer
    )
    get_parameter_net(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainLoader, testLoader = load_data(path=ABSOLUTE_PATH, batchsize=args.batchsize, tag=args.gap, device=device)
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
