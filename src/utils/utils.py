from matplotlib import pyplot as plt
import torch
import numpy as np
import yaml
import argparse
from datetime import datetime
import os
import sys
from termcolor import colored
import pdb
from sklearn import gaussian_process as gp

COLOR_LIST = [
    "b",
    "r",
    "g",
    "y",
    "c",
    "m",
    "skyblue",
    "indigo",
    "goldenrod",
    "salmon",
    "pink",
    "silver",
    "darkgreen",
    "lightcoral",
    "navy",
    "orchid",
    "steelblue",
    "saddlebrown",
    "orange",
    "olive",
    "tan",
    "firebrick",
    "maroon",
    "darkslategray",
    "crimson",
    "dodgerblue",
    "aquamarine",
    "b",
    "r",
    "g",
    "y",
    "c",
    "m",
    "skyblue",
    "indigo",
    "goldenrod",
    "salmon",
    "pink",
    "silver",
    "darkgreen",
    "lightcoral",
    "navy",
    "orchid",
    "steelblue",
    "saddlebrown",
    "orange",
    "olive",
    "tan",
    "firebrick",
    "maroon",
    "darkslategray",
    "crimson",
    "dodgerblue",
    "aquamarine",
]


# basic utils
class Printer(object):
    ## Example, to print code running time between two p.print() calls
    # p.print(f"test_start", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """

        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(
        self,
        item,
        tabs=0,
        is_datetime=None,
        banner_size=0,
        end=None,
        avg_window=-1,
        precision="second",
        is_silent=False,
    ):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2],
                avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window + 1, len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, "yellow"))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))


def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time

    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string


# config utils
def add_args_from_config(parser):
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    existing_args = {action.dest for action in parser._actions}

    # Add arguments from the config file to the parser
    for key, value in config.items():
        # If the argument is not already added, add it to the parser
        if key not in existing_args:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    return parser


def save_config_from_args(args, config_dir):
    config_dict = {k: v for k, v in vars(args).items() if k != "config"}
    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # config_dir = args.exp_path + "/results/" + time_now
    # os.makedirs(config_dir, exist_ok=True)  # Ensure the directory exists
    config_file_path = os.path.join(config_dir, "config.yaml")
    with open(config_file_path, "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False)

    return


# result analysis
def caculate_confidence_interval(data):
    """'
    input example: abs(pred_design-pred_simu)
    """
    list_dim = range(data.dim())
    if data.dim() > 1:
        MAE_batch_size = torch.mean(data, dim=tuple(list_dim[1:]))
    else:
        MAE_batch_size = data
    mean = torch.mean(MAE_batch_size)

    std_dev = torch.std(MAE_batch_size)
    min_value = min(MAE_batch_size)
    confidence_level = 0.95
    # pdb.set_trace()
    n = len(data)
    # kk = stats.t.ppf((1 + confidence_level) / 2, n - 1) * (std_dev / (n ** 0.5))
    margin_of_error = std_dev * 1.96 / torch.sqrt(torch.tensor(n, dtype=float))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    print("mean:", mean.item())
    print("std:", std_dev.item())
    print(f"margin_of_error:", margin_of_error)

    return mean, std_dev, margin_of_error, min_value


# training utils
def caculate_num_parameters(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # pdb.set_trace()
    return


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


def create_res(path: str, folder_name: str):
    """create result folder

    Args:
        path (str): Overall results folder
        folder_name (str): subfolder in results folder
    """
    if not os.path.exists(path):
        os.makedirs(path)
    res_path = os.path.join(path, folder_name)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        print(f"{res_path} is created。")
    else:
        print(f"{res_path} had been created.")
    return res_path


def get_parameter_net(net):
    total_num = sum(p.numel() for p in net.parameters())
    train_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"the parameter of the net is {train_num:d}")
    return total_num, train_num


def find_max_min(array):
    """
    This function accepts a numpy array or a torch tensor and returns the maximum and minimum values.

    Parameters:
    array (np.ndarray or torch.Tensor): The array or tensor to be checked.

    Returns:
    tuple: A tuple containing the maximum and minimum values.
    """
    # Check if the input is a numpy array
    if isinstance(array, np.ndarray):
        max_val = np.max(array)
        min_val = np.min(array)
    # Check if the input is a torch tensor
    elif isinstance(array, torch.Tensor):
        max_val = torch.max(array).item()  # .item() is used to convert the tensor to a scalar
        min_val = torch.min(array).item()
    else:
        # Raise an error if the input is not a numpy array or a torch tensor
        raise TypeError("Input must be a numpy array or a torch tensor")

    return max_val, min_val


class GRF(object):
    """generate 1D gaussian random field

    Args:
        object (_type_): _description_
    """

    def __init__(self, T=1, kernel="RBF", mean=0.0, length_scale=1.0, sigma=1, N=1000, interp="cubic"):
        """_summary_

        Args:
            T (int, optional): _description_. Defaults to 1.
            kernel (str, optional): _description_. Defaults to "RBF".
            mean (float, optional): mean. Defaults to 0.0.
            length_scale (float, optional): smooth factor. Defaults to 1.0.
            sigma (int, optional): standard deviation. Defaults to 1.
            N (int, optional): _description_. Defaults to 1000.
            interp (str, optional): _description_. Defaults to "cubic".
        """
        self.N = N
        self.mean = mean
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale) * sigma**2
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n):
        """Generate `n` random feature vectors."""
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T + self.mean


def plot_compare_2d(true_d, pred_d, savep=None, paraname="", Unit_="", language="eng"):
    # compare
    # shape: w*h
    if torch.is_tensor(true_d):
        if true_d.is_cuda:
            true_d = true_d.cpu()
            pred_d = pred_d.cpu()
        true_d = true_d.numpy()
        pred_d = pred_d.numpy()
    title_list = ["True ", "Pred ", "Error "]

    re_err = np.abs((true_d - pred_d) / (true_d + 1e-6))
    err = true_d - pred_d
    plt.figure(figsize=(8, 6), dpi=100)

    Unit1 = ""
    if len(Unit_) > 0:
        Unit1 = "(" + Unit_ + ")"

    plt.subplot(131)
    plt.imshow(
        true_d,
        cmap="jet",
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        interpolation="bilinear",
    )
    plt.title(title_list[0] + paraname + Unit1, fontsize=15)
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(
        pred_d,
        cmap="jet",
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        interpolation="bilinear",
    )

    plt.title(title_list[1] + paraname + Unit1, fontsize=15)
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(
        err,
        cmap="jet",
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        interpolation="bilinear",
    )

    plt.title(title_list[2] + paraname + Unit1, fontsize=15)
    plt.colorbar()

    plt.tight_layout()
    if savep is not None:
        plt.savefig(savep)
    plt.show()


def plot_scatter_compare(
    coord_x,
    coord_y,
    true_d,
    pred_d,
    pointsize=1,
    figsize=(18, 4),
    savep=None,
    fontsize=15,
    cmap="viridis",
    paraname="",
    Unit_="",
    language="eng",
    e_min=None,
    e_max=None,
):
    # compare
    # shape: w*h
    if torch.is_tensor(true_d):
        if true_d.is_cuda:
            true_d = true_d.cpu()
            pred_d = pred_d.cpu()
        true_d = true_d.numpy()
        pred_d = pred_d.numpy()
    title_list = ["Ground truth ", "Predition ", "Error "]

    vmin = min(np.min(true_d), np.min(pred_d))
    vmax = max(np.max(true_d), np.max(pred_d))

    Unit1 = ""
    if len(Unit_) > 0:
        Unit1 = "(" + Unit_ + ")"
    plt.figure(figsize=figsize)
    plt.subplot(131)
    sc1 = plt.scatter(coord_x, coord_y, c=true_d, cmap=cmap, edgecolors=None, s=pointsize, vmin=vmin, vmax=vmax)
    plt.title(title_list[0] + paraname + Unit1, fontsize=fontsize)
    plt.colorbar(sc1)

    plt.subplot(132)
    sc2 = plt.scatter(coord_x, coord_y, c=pred_d, cmap=cmap, edgecolors=None, s=pointsize, vmin=vmin, vmax=vmax)
    plt.title(title_list[1] + paraname + Unit1, fontsize=fontsize)
    plt.colorbar(sc2)

    plt.subplot(133)
    if e_max is None or e_min is None:
        e_max, e_min = find_max_min(pred_d - true_d)
    sc3 = plt.scatter(
        coord_x, coord_y, c=(pred_d - true_d), cmap=cmap, edgecolors=None, s=pointsize, vmin=e_min, vmax=e_max
    )
    plt.title(title_list[2] + paraname + Unit1, fontsize=fontsize)
    plt.colorbar(sc3)

    plt.tight_layout()
    if savep is not None:
        plt.savefig(savep)
    plt.show()


def plot_contourf(Z, savep, xlabel="x", ylabel="y", title=None):
    plt.imshow(
        Z,
        cmap="jet",
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        interpolation="bilinear",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(savep)
    plt.close()


def random_split_line(n, bound, min_gap):
    # n: points
    d_step = 1 / n
    while True:
        divisions = np.linspace(0, 1, n)
        divisions = np.random.uniform(0, d_step, n - 1) + divisions[:-1]
        divisions = (divisions - divisions[0]) * (bound[1] - bound[0]) + bound[0]
        divisions = np.concatenate((divisions, np.array(bound[1]).reshape(-1)))
        d_divisions = divisions[1:] - divisions[:-1]
        if np.min(d_divisions) >= min_gap:
            break
    return divisions


def minmax_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def L2_norm(array):
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach().numpy()
    b = array.shape[0]
    array = array.reshape(b, -1)
    norm = np.sum(array**2, axis=1) ** 0.5
    return norm


def relative_error(data_t, data_p):
    return np.sum(L2_norm(data_t - data_p) / L2_norm(data_t)) / data_t.shape[0]


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x
