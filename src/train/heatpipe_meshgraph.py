import sys, os
import argparse
import torch
import numpy as np
from torch_geometric.data import Data as Data_g
from torch_geometric.data import DataLoader as DataLoader_G

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from filepath import ABSOLUTE_PATH

sys.path.append(ABSOLUTE_PATH)

from src.train.train import Trainer
from src.model.meshgraphnet import EncoderProcesserDecoder as MeshGraphNet
from src.utils.utils import create_res, set_seed, get_time, save_config_from_args, get_parameter_net, find_max_min
import time


def get_loss(model: MeshGraphNet, batch, loss_fn):
    outputs_p = model(batch)
    loss = loss_fn(batch.y, outputs_p)
    return loss


def load_data(path, batchsize, tag, device):
    x = torch.tensor(np.load(path + "/data/heatpipe/x_element_as_one_node.npy")).to(device).float()
    y = torch.tensor(np.load(path + "/data/heatpipe/y_element_as_one_node.npy")).to(device).float()
    edge_index = np.load(path + "/data/heatpipe/adj_element_as_one_node.npy").transpose(0, 2, 1)
    # print(np.max(edge_index), np.min(edge_index))
    edge_index = torch.tensor(edge_index).to(device).long()

    train_dataset = []

    b, n_node, n_dim = x.shape
    n_edge = edge_index.shape[-1]
    edge_attr = torch.zeros((b, n_edge, 1), device=device)
    for i in range(tag):
        data = Data_g(x=x[i], y=y[i], edge_index=edge_index[i], edge_attr=edge_attr[i])
        train_dataset.append(data)
    train_loader = DataLoader_G(train_dataset, batch_size=batchsize, shuffle=True)
    test_dataset = []
    for i in range(tag, x.shape[0]):
        data = Data_g(x=x[i], y=y[i], edge_index=edge_index[i], edge_attr=edge_attr[i])
        test_dataset.append(data)
    test_loader = DataLoader_G(test_dataset, batch_size=batchsize * 10, shuffle=True)
    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train simple distribution model")
    parser.add_argument("--exp_id", default="heatpipe", type=str, help="experiment folder id")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--batchsize", default=256, type=int, help="size of dataset")
    parser.add_argument("--epoches", default=100000, type=int, help="training epoch")
    parser.add_argument("--checkpoint", default=2000, type=int, help="save and sample period")
    parser.add_argument("--overall_results_path", default="results", type=str, help="where to create result folder")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--node_input_size", default=4, type=int, help="in dim")
    parser.add_argument("--edge_input_size", default=1, type=int, help="in dim")
    parser.add_argument("--message_passing_num", default=6, type=int, help="output dim")
    parser.add_argument("--hidden_size", default=256, type=int, help="hidden dim")
    parser.add_argument("--out_dim", default=2412, type=int, help="output dim")
    parser.add_argument("--gradient_accumulate_every", default=2, type=int, help="gradient_accumulate_every")
    parser.add_argument("--gap", default=800, type=int, help="dataset size for train")
    parser.add_argument("--model_type", default="meshgraph_6_256", type=str, help="gnn or transformer")
    parser.add_argument("--paradigm", default="surrogate", type=str, help="diffusion or surrogate")
    args = parser.parse_args()
    set_seed(args.seed)
    results_path = create_res(path=os.path.join(ABSOLUTE_PATH, args.overall_results_path), folder_name=args.exp_id)
    model_type = args.model_type
    paradigm = args.paradigm
    results_folder = os.path.join(results_path, paradigm, model_type)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    model = MeshGraphNet(
        args.message_passing_num,
        args.node_input_size,
        args.edge_input_size,
        hidden_size=args.hidden_size,
        output_size=args.out_dim,
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
