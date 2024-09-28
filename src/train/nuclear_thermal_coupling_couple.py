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
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from functools import partial

# import path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from filepath import ABSOLUTE_PATH

sys.path.append(ABSOLUTE_PATH)
from src.model.diffusion import GaussianDiffusion
from src.train.train import Trainer
from src.model.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D, MLP
from src.model.fno import FNO3D
from src.utils.utils import create_res, set_seed, get_time, save_config_from_args, get_parameter_net, find_max_min
import time


def load_nt_dataset_emb(field="neutron", n_data_set=None, device="cuda"):
    folder_path = ABSOLUTE_PATH + "/data/NTcouple/couple"
    cond = torch.tensor(np.load(folder_path + "/bc.npy")[0:n_data_set]).float().to(device)  # b,1,t,ny,1
    cond = normalize(cond, "neutron")
    if field == "neutron":
        data = torch.tensor(np.load(folder_path + "/neu.npy")[0:n_data_set]).float().to(device)
        data = normalize(data, "neutron")
    elif field == "solid":
        data = torch.tensor(np.load(folder_path + "/fuel.npy")[0:n_data_set]).float().to(device)  # b,1,t,ny,nx
        data = normalize(data, "solid")
    elif field == "fluid":
        data = torch.tensor(np.load(folder_path + "/fluid.npy")[0:n_data_set]).float().to(device)  # b,4,t,ny,nx
        data = normalize(data, "fluid")
    ny = data.shape[-1]
    cond = cond.expand(-1, -1, -1, -1, ny)
    cond_lis = [cond]
    return cond_lis, data


def cond_emb():

    def consistent(x):
        return x

    return [consistent]


def forward_function(paradigm):
    if paradigm == "surrogate":

        def func(model: nn.Module, batch, loss_fn):
            data, *cond = batch
            outputs_p = model(cond)
            loss = loss_fn(data, outputs_p)
            return loss

        return func, func
    elif paradigm == "diffusion":

        def func_train(model: nn.Module, batch, loss_fn=F.mse_loss):
            data, *cond = batch
            loss = model(data, cond)
            return loss

        def func_val(model: nn.Module, batch, loss_fn=F.mse_loss):
            data, *cond = batch
            batchsize = data.shape[0]
            outputs_p = model.sample(batchsize, cond)
            loss = loss_fn(data, outputs_p)
            return loss

        return func_train, func_val
    else:
        raise Exception("paradigm is not exist")


def normalize(x, field):
    solid_bound = [100, 1500.0]
    flux_bound = [-2000.0, 4000]
    fluid_bound = [[100, 1200.0], [-50.0, 250.0], [-0.006, 0.006], [0.0, 0.6]]
    n_bound = [0, 3.258]  # [0, 25.0]
    if field == "solid":
        n_x = (x - solid_bound[0]) / (solid_bound[1] - solid_bound[0]) * 2 - 1
    elif field == "fluid":
        for i in range(x.shape[1]):
            x[:, i] = (x[:, i] - fluid_bound[i][0]) / (fluid_bound[i][1] - fluid_bound[i][0]) * 2 - 1
        n_x = x
    elif field == "neutron":
        n_x = (torch.log(x + 1) - n_bound[0]) / (n_bound[1] - n_bound[0]) * 2 - 1
    elif field == "flux":
        n_x = (x - flux_bound[0]) / (flux_bound[1] - flux_bound[0]) * 2 - 1
    # print(torch.max(n_x), torch.min(n_x))
    return n_x


def renormalize(x, field):
    solid_bound = [100, 1500.0]
    flux_bound = [-2000.0, 4000]
    fluid_bound = [[100, 1200.0], [-50.0, 250.0], [-0.006, 0.006], [0.0, 0.6]]
    n_bound = [0, 3.258]  # [0, 25.0]
    if field == "solid":
        rn_x = (x + 1) * 0.5 * (solid_bound[1] - solid_bound[0]) + solid_bound[0]
    elif field == "fluid":
        for i in range(x.shape[1]):
            x[:, i] = (x[:, i] + 1) * 0.5 * (fluid_bound[i][1] - fluid_bound[i][0]) + fluid_bound[i][0]
        rn_x = x
    elif field == "neutron":
        rn_x = torch.exp((x + 1) * 0.5 * (n_bound[1] - n_bound[0]) + n_bound[0]) - 1
    elif field == "flux":
        rn_x = (x + 1) * 0.5 * (flux_bound[1] - flux_bound[0]) + flux_bound[0]
    # print(torch.max(rn_x), torch.min(rn_x))
    else:
        print("field name wrong")
    return rn_x


if __name__ == "__main__":
    # field = "fluid"
    # b = 20
    # cond, data = load_nt_dataset_emb(field, device="cpu")
    # for i in range(len(cond)):
    #     cond[i] = cond[i][:b]
    # data = data[:b]
    # t = torch.rand(b)
    # emb = cond_emb(field)
    # UNet = Unet3D_with_Conv3D(dim=16, cond_dim=1, out_dim=4, cond_emb=emb, dim_mults=(1, 2, 4))
    # get_parameter_net(UNet)
    # out = UNet(data, t, cond)
    # print(out.shape)

    parser = argparse.ArgumentParser(description="Train simple distribution model")
    parser.add_argument("--exp_id", default="nuclear_thermal_coupling_couple", type=str, help="experiment folder id")
    parser.add_argument("--train_which", default="fluid", type=str, help="train neutron or solid or fluid")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--batchsize", default=32, type=int, help="size of dataset")
    parser.add_argument("--epoches", default=100000, type=int, help="training epoch")
    parser.add_argument("--diffusion_step", default=250, type=int, help="diffusion_step")
    parser.add_argument("--checkpoint", default=2000, type=int, help="save and sample period")
    parser.add_argument("--overall_results_path", default="results", type=str, help="where to create result folder")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--dim", default=8, type=int, help="encode dim")
    parser.add_argument("--dataset", default="iter1", type=str, help="dataset, option: iter1, iter2")
    parser.add_argument("--n_dataset", default=5000, type=int, help="n of data")
    # parser.add_argument("--out_dim", default=9, type=int, help="cond channel")
    parser.add_argument("--gap", default=1000, type=int, help="dataset size for valiadation")
    parser.add_argument("--model_type", default="Unet", type=str, help="Unet or ViT or FNO")
    parser.add_argument("--paradigm", default="diffusion", type=str, help="diffusion or surrogate")
    parser.add_argument("--gradient_accumulate_every", default=2, type=int, help="gradient_accumulate_every")
    # ViT
    parser.add_argument("--depth", default="2", type=int, help="transformer depth")
    parser.add_argument("--head", default="8", type=int, help="head of attention")
    parser.add_argument("--vit_dim", default="256", type=int, help="vit dim")
    parser.add_argument("--mlp_dim", default="256", type=int, help="vit mlp dim")
    parser.add_argument("--patchsize", default=(2, 8, 2), type=tuple, help="patchsize")
    # FNO
    parser.add_argument("--fno_nlayer", default=2, type=int, help="fno layers")
    parser.add_argument("--fno_layer_size", default=8, type=int, help="fno_layer_size")
    parser.add_argument(
        "--fno_modes", default="6,16,6", type=lambda s: [int(item) for item in s.split(",")], help="List of FNO modes"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    results_path = create_res(path=os.path.join(ABSOLUTE_PATH, args.overall_results_path), folder_name=args.exp_id)
    train_which = args.train_which
    paradigm = args.paradigm
    model_type = args.model_type
    results_folder = os.path.join(
        results_path, paradigm + model_type + train_which, args.dataset + "_" + str(args.n_dataset)
    )
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    save_config_from_args(args, results_folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cond, data = load_nt_dataset_emb(train_which, args.n_dataset, device=device)
    emb = cond_emb()
    interval = -args.gap

    image_size = data.shape[-2:]
    frames = data.shape[2]
    train_dataset = TensorDataset(data[:interval], cond[0][:interval])
    data_val = TensorDataset(data[interval:], cond[0][interval:])
    train_function, val_function = forward_function(paradigm)
    if paradigm == "diffusion":
        if model_type == "Unet":
            model = Unet3D_with_Conv3D(
                dim=args.dim,
                cond_dim=len(emb),
                out_dim=data.shape[1],
                cond_emb=emb,
                dim_mults=(1, 2, 4),
                use_sparse_linear_attn=False,
                attn_dim_head=16,
            ).to(device)
        elif model_type == "ViT":
            model = ViT(
                image_size=image_size,
                image_patch_size=args.patchsize[1:],
                frames=frames,
                frame_patch_size=args.patchsize[0],
                dim=args.vit_dim,
                depth=args.depth,
                heads=args.head,
                mlp_dim=args.mlp_dim,
                cond_emb=emb,
                Time_Input=True,
                channels=len(emb) + data.shape[1],
                out_channels=data.shape[1],
                dropout=0.0,
                emb_dropout=0.0,
            )
        elif model_type == "FNO":
            model = FNO3D(
                in_channels=len(emb) + data.shape[1],
                out_channels=data.shape[1],
                nr_fno_layers=args.fno_nlayer,
                fno_layer_size=args.fno_layer_size,
                fno_modes=args.fno_modes,
                time_input=True,
                cond_emb=emb,
            )
        diffusion = GaussianDiffusion(
            model, seq_length=tuple(data.shape[1:]), timesteps=args.diffusion_step, auto_normalize=False
        ).to(device)
        # diffusion.load_state_dict(
        #     torch.load(
        #         "../../results/nuclear_thermal_coupling/diffusion"
        #         + model_type
        #         + train_which
        #         + "/iter1_5000/model-50.pt"
        #     )["model"]
        # )
        get_parameter_net(diffusion)
        train = Trainer(
            model=diffusion,
            data_train=train_dataset,
            train_function=train_function,
            val_function=val_function,
            data_val=data_val,
            train_lr=args.lr,
            train_num_steps=args.epoches,
            train_batch_size=args.batchsize,
            save_every=args.checkpoint,
            results_folder=results_folder,
            gradient_accumulate_every=args.gradient_accumulate_every,
        )

        train.train()
    elif paradigm == "surrogate":
        if model_type == "Unet":
            model = Unet3D_with_Conv3D(
                dim=args.dim,
                cond_dim=0,
                channels=len(emb),
                out_dim=data.shape[1],
                cond_emb=emb,
                time_cond=False,
                dim_mults=(1, 2, 4),
                use_sparse_linear_attn=False,
                attn_dim_head=16,
            ).to(device)
        elif model_type == "ViT":
            model = ViT(
                image_size=image_size,
                image_patch_size=args.patchsize[1:],
                frames=frames,
                frame_patch_size=args.patchsize[0],
                dim=args.vit_dim,
                depth=args.depth,
                heads=args.head,
                mlp_dim=args.mlp_dim,
                cond_emb=emb,
                channels=len(emb),
                out_channels=data.shape[1],
                dropout=0.0,
                emb_dropout=0.0,
            )
        elif model_type == "FNO":
            model = FNO3D(
                in_channels=len(emb),
                out_channels=data.shape[1],
                nr_fno_layers=args.fno_nlayer,
                fno_layer_size=args.fno_layer_size,
                fno_modes=args.fno_modes,
                time_input=False,
                cond_emb=emb,
            )
        get_parameter_net(model)
        train = Trainer(
            model=model,
            data_train=train_dataset,
            train_lr=args.lr,
            data_val=data_val,
            train_function=train_function,
            val_function=val_function,
            train_num_steps=args.epoches,
            train_batch_size=args.batchsize,
            save_every=args.checkpoint,
            results_folder=results_folder,
            gradient_accumulate_every=args.gradient_accumulate_every,
        )
        train.train()
