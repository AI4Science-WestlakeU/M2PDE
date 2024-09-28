import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from accelerate import Accelerator
from ema_pytorch import EMA
from torch.optim import Adam
from pathlib import Path
from tqdm.auto import tqdm
import os
import sys
import yaml
from torch.optim.lr_scheduler import StepLR
from functools import partial


# import path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from filepath import ABSOLUTE_PATH

sys.path.append(ABSOLUTE_PATH)
from src.train.train import Trainer
from src.model.transolver import Transolver
from src.model.GeoFNO import GeoFNO2d as FNO
from src.model.diffusion import GaussianDiffusion
from src.utils.utils import create_res, set_seed, get_time, save_config_from_args, get_parameter_net, find_max_min
import time
from torch.utils.data import DataLoader, TensorDataset


def load_data(path, batchsize, tag, device, model_type="transformer"):
    if model_type == "transformer":
        # def load_data(path, batchsize, tag, device):
        x = torch.tensor(np.load(path + "/data/heatpipe/x.npy")).to(device).float()  # b, 804, 10
        y = torch.tensor(np.load(path + "/data/heatpipe/y.npy")).to(device).float()  # b, 804, 3
        coord = torch.tensor(np.load(path + "/data/heatpipe/coord.npy")).to(device).float()
        coord[:, 0] = (coord[:, 0] - 0.0455) / (0.065345 - 0.0455) * 2 - 1
        coord[:, 1] = (coord[:, 1] - 0.072) / (0.08918 - 0.072) * 2 - 1
        coord = coord.expand(x.shape[0], -1, -1)

        train_dataset = TensorDataset(coord[:tag], x[:tag], y[:tag])
        test_dataset = TensorDataset(coord[tag:], x[tag:], y[tag:])
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batchsize * 10, shuffle=True)
        return train_loader, test_loader
    elif model_type == "FNO":
        # def load_data(path, batchsize, tag, device):
        x = torch.tensor(np.load(path + "/data/heatpipe/x.npy")).to(device).float()  # b, 804, 10
        y = torch.tensor(np.load(path + "/data/heatpipe/y.npy")).to(device).float()  # b, 804, 3

        coord = torch.zeros(x.shape[1], 3).to(device)
        coordxy = torch.tensor(np.load(path + "/data/heatpipe/coord.npy")).to(device).float()
        coordxy[:, 0] = (coordxy[:, 0] - 0.0455) / (0.065345 - 0.0455)
        coordxy[:, 1] = (coordxy[:, 1] - 0.072) / (0.08918 - 0.072)
        coord[:, :-1] = coordxy
        coord = coord.expand(x.shape[0], -1, -1)

        train_dataset = TensorDataset(coord[:tag], x[:tag], y[:tag])
        test_dataset = TensorDataset(coord[tag:], x[tag:], y[tag:])
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batchsize * 10, shuffle=True)
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


def forward_function(model_type, paradigm):
    if paradigm == "surrogate":
        if model_type in ["transformer", "FNO"]:

            def func(model: nn.Module, batch, loss_fn):
                # batchsize = torch.max(batch.batch) + 1
                # coord, fx = batch.coord.reshape(batchsize, -1, 2), batch.x.reshape(batchsize, -1, batch.x.shape[-1])
                # outputs_p = model(data=(coord, fx)).reshape(-1, batch.y.shape[-1])
                coord, fx, y = batch
                outputs_p = model(data=(coord, fx))
                loss = loss_fn(y, outputs_p)
                return loss

            return func, func
        else:
            raise Exception("model type is not exist")
    elif paradigm == "diffusion":
        if model_type in ["transformer", "FNO"]:

            def func_train(model: nn.Module, batch, loss_fn=F.mse_loss):
                # batchsize = torch.max(batch.batch) + 1
                # coord, fx = batch.coord.reshape(batchsize, -1, 2), batch.x.reshape(batchsize, -1, batch.x.shape[-1])
                # img = batch.y.reshape(batchsize, -1, batch.y.shape[-1])
                coord, fx, y = batch
                loss = model(y, (coord, fx))
                return loss

            def func_val(model: nn.Module, batch, loss_fn=F.mse_loss):
                # batchsize = torch.max(batch.batch) + 1
                # coord, fx = batch.coord.reshape(batchsize, -1, 2), batch.x.reshape(batchsize, -1, batch.x.shape[-1])
                coord, fx, y = batch
                batchsize = y.shape[0]
                outputs_p = model.sample(batchsize, (coord, fx))
                loss = loss_fn(y, outputs_p)
                return loss

            return func_train, func_val
        else:
            raise Exception("model type is not exist")
    else:
        raise Exception("paradigm is not exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train simple distribution model")
    parser.add_argument("--exp_id", default="heatpipe", type=str, help="experiment folder id")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--batchsize", default=64, type=int, help="size of dataset")
    parser.add_argument("--epoches", default=100000, type=int, help="training epoch")
    parser.add_argument("--diffusion_step", default=250, type=int, help="diffusion_step")
    parser.add_argument("--checkpoint", default=2000, type=int, help="save and sample period")
    parser.add_argument("--overall_results_path", default="results", type=str, help="where to create result folder")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--out_dim", default=3, type=int, help="output dim")
    parser.add_argument("--hidden_dim", default=96, type=int, help="hidden dim")
    parser.add_argument("--cond_dim", default=10, type=int, help="input dim")
    parser.add_argument("--n_layer", default=3, type=int, help="layers")
    parser.add_argument("--slice_num", default=16, type=int, help="transolver slice_num")
    parser.add_argument("--num_node", default=804, type=int, help="num_node")
    parser.add_argument("--gradient_accumulate_every", default=2, type=int, help="gradient_accumulate_every")
    parser.add_argument("--gap", default=14000, type=int, help="dataset size for train")
    parser.add_argument("--model_type", default="FNO", type=str, help="gnn or transformer or fno")
    parser.add_argument("--paradigm", default="surrogate", type=str, help="diffusion or surrogate")
    # fno
    parser.add_argument("--fno_layer_size", default=32, type=int, help="fno_layer_size")
    parser.add_argument(
        "--fno_modes", default="16,16,16", type=lambda s: [int(item) for item in s.split(",")], help="List of FNO modes"
    )

    args = parser.parse_args()
    set_seed(args.seed)
    results_path = create_res(path=os.path.join(ABSOLUTE_PATH, args.overall_results_path), folder_name=args.exp_id)
    model_type = args.model_type
    paradigm = args.paradigm
    results_folder = os.path.join(results_path, paradigm, model_type)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    save_config_from_args(args, results_folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainLoader, testLoader = load_data(
        path=ABSOLUTE_PATH, batchsize=args.batchsize, tag=args.gap, device=device, model_type=model_type
    )
    # test_dataset = TensorDataset(data[interval:], cond[interval:])
    train_function, val_function = forward_function(model_type=model_type, paradigm=paradigm)
    if paradigm == "diffusion":
        if model_type == "transformer":
            model = Transolver(
                space_dim=2,
                n_layers=args.n_layer,
                n_hidden=args.hidden_dim,
                dropout=0.0,
                n_head=8,
                Time_Input=True,
                act="gelu",
                mlp_ratio=1,
                fun_dim=args.cond_dim + args.out_dim,
                out_dim=args.out_dim,
                slice_num=args.slice_num,
                ref=8,
                unified_pos=False,
            )
            diffusion = GaussianDiffusion(
                model,
                seq_length=tuple([args.num_node, args.out_dim]),
                timesteps=args.diffusion_step,
                auto_normalize=False,
            ).to(device)
        elif model_type == "FNO":
            modes = args.fno_modes
            model = FNO(
                modes1=modes[0],
                modes2=modes[1],
                modes3=modes[2],
                width=args.fno_layer_size,
                in_channels=args.cond_dim + args.out_dim,
                out_channels=args.out_dim,
                time_input=True,
            ).to(device)
            diffusion = GaussianDiffusion(
                model,
                seq_length=tuple([args.num_node, args.out_dim]),
                timesteps=args.diffusion_step,
                auto_normalize=False,
            ).to(device)
        get_parameter_net(diffusion)
        train = Trainer(
            model=diffusion,
            data_train=trainLoader,
            data_val=testLoader,
            train_function=train_function,
            val_function=val_function,
            train_lr=args.lr,
            train_num_steps=args.epoches,
            train_batch_size=args.batchsize,
            save_every=args.checkpoint,
            results_folder=results_folder,
            gradient_accumulate_every=args.gradient_accumulate_every,
        )

        train.train()
    elif paradigm == "surrogate":
        if model_type == "transformer":
            model = Transolver(
                space_dim=2,
                n_layers=args.n_layer,
                n_hidden=args.hidden_dim,
                dropout=0.0,
                n_head=8,
                Time_Input=True,
                act="gelu",
                mlp_ratio=1,
                fun_dim=args.cond_dim,
                out_dim=args.out_dim,
                slice_num=args.slice_num,
                ref=8,
                unified_pos=False,
            ).to(device)
        elif model_type == "FNO":
            modes = args.fno_modes
            model = FNO(
                modes1=modes[0],
                modes2=modes[1],
                modes3=modes[2],
                width=args.fno_layer_size,
                in_channels=args.cond_dim,
                out_channels=args.out_dim,
            ).to(device)
        get_parameter_net(model)
        train = Trainer(
            model=model,
            data_train=trainLoader,
            data_val=testLoader,
            train_function=train_function,
            val_function=val_function,
            train_lr=args.lr,
            gradient_accumulate_every=args.gradient_accumulate_every,
            train_num_steps=args.epoches,
            train_batch_size=args.batchsize,
            save_every=args.checkpoint,
            results_folder=results_folder,
        )
        train.train()
