import os
import random
import numpy as np
import subprocess
from tqdm.auto import tqdm
import argparse

dis_center = 16
initial_x = 13.8564
initial_y = 8.0
coord = np.array(
    [
        [initial_x, initial_y],
        [initial_x + 1 * dis_center * 3**0.5, initial_y],
        [initial_x + 2 * dis_center * 3**0.5, initial_y],
        [initial_x + 3 * dis_center * 3**0.5, initial_y],
        [initial_x + 3**0.5 / 2 * dis_center, initial_y + dis_center * 1.5],
        [initial_x + 3**0.5 / 2 * dis_center + 1 * dis_center * 3**0.5, initial_y + dis_center * 1.5],
        [initial_x + 3**0.5 / 2 * dis_center + 2 * dis_center * 3**0.5, initial_y + dis_center * 1.5],
        [initial_x + 3**0.5 * dis_center + 0 * dis_center * 3**0.5, initial_y + dis_center * 3],
        [initial_x + 3**0.5 * dis_center + 1 * dis_center * 3**0.5, initial_y + dis_center * 3],
        [initial_x + dis_center * 3**0.5 * 1.5, initial_y + dis_center * 4.5],
        [initial_x + 3**0.5 / 2 * dis_center, initial_y + 1 / 2 * dis_center],
        [initial_x + 3**0.5 / 2 * dis_center + 1 * dis_center * 3**0.5, initial_y + 1 / 2 * dis_center],
        [initial_x + 3**0.5 / 2 * dis_center + 2 * dis_center * 3**0.5, initial_y + 1 / 2 * dis_center],
        [initial_x + 3**0.5 * dis_center, initial_y + 2 * dis_center],
        [initial_x + 3**0.5 * dis_center * 2, initial_y + 2 * dis_center],
        [initial_x + dis_center * 3**0.5 * 1.5, initial_y + dis_center * 3.5],
    ]
)
coord_x = np.array(
    [
        13.8564,
        27.7128064605510,
        41.569212921102036,
        55.42561938165305,
        69.28202584220406,
        83.1384323027550,
        96.99483876330612,
    ]
)
coord_y = np.array([8.0, 16, 32, 40, 56, 64])


def gen_flux(l_b, u_b):
    flux = []
    for i in range(len(coord)):
        flux.append(round(np.random.uniform(l_b, u_b) * 1e6, 0))
    flux_ = np.zeros((7, 6))
    for i in range(len(coord)):
        ind_x = np.argmin(np.abs(coord[i, 0] - coord_x))
        ind_y = np.argmin(np.abs(coord[i, 1] - coord_y))
        flux_[ind_x, ind_y] = flux[i]
    return flux_


def gen_flux1(l_b, u_b):
    flux = []
    for i in range(len(coord)):
        flux.append(round(np.random.uniform(l_b, u_b) * 1e6, 0))
    return flux


def replacement(Z):
    data = dict()
    for i in range(len(coord_y)):
        for j in range(len(coord_x)):
            data += "%.1f " % Z[j, i]
    return {"data": data}


def replacement_inp(flux):
    replace = dict()
    for i in range(len(flux)):
        key = "b" + str(i + 1)
        replace[key] = flux[i]
    return replace


def write_inp(base_file, out_file, replacements):
    #'./inpbase.txt' './neutroninp/phi.txt fluidT.txt fuelT.txt'
    with open(base_file, "r", encoding="utf-8") as base:
        template = base.read()
    src = template % replacements
    with open(out_file, "w", encoding="utf-8") as file:
        file.write(src)


sys_b_sub = """
    [./%(b)s]
      boundary = %(b)s
      penalty = 1e15
      displacements = "disp_x disp_y"
    [../]
"""
sys_b = """
  [./InclinedNoDisplacementBC]
    %(module)s
  [../]"""
dir_b = """
  [./%(b)s_y]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = %(b)s
    function = fix_bc_y
  [../]
  [./%(b)s_x]
    type = FunctionDirichletBC
    variable = disp_x
    boundary = %(b)s
    function = fix_bc_x
  [../]
"""

function = """
    [./fix_bc_x]
        type = PiecewiseMultilinear
        data_file = disp_x.txt
    [../]
    [./fix_bc_y]
        type = PiecewiseMultilinear
        data_file = disp_y.txt
    [../]
"""
ic = """
    [ICs]
        [./ic_disp_x]
            type = FunctionIC
            variable = 'disp_x'
            function = U_IC
        [../]
        [./ic_disp_y]
            type = FunctionIC
            variable = 'disp_y'
            function = U_IC
        [../]
    []
"""
ic_flux = """
  [./ic_flux_%(block)s]
    type = ConstantIC
    variable = flux_BC
    value = %(flux)s
    block = %(block)s
  [../]
"""


Free = "\n"


def gen_boundary(base_file="./represent_base.i", out_file="./represent.i"):

    disp_left, disp_right = 0e-4, 8e-4
    d_disp_l, d_disp_h = 0, 0
    disp_xl = np.random.uniform(disp_left, disp_right)
    d_disp_x = np.random.uniform(d_disp_l, d_disp_h)
    disp_xh = disp_xl + d_disp_x
    disp_yl = np.random.uniform(disp_left, disp_right)
    d_disp_y = np.random.uniform(d_disp_l, d_disp_h)
    disp_yh = disp_yl + d_disp_y
    disp_x_b = np.linspace(disp_xl, disp_xh, 4)
    disp_y_b = np.linspace(disp_yl, disp_yh, 4)
    coord_x_all = [
        [0.0138564, 0.0415692, 0.069282, 0.0969948],
        [0.006928200, 0.0207846, 0.034641, 0.0484974],
        [0.0623538, 0.0762102, 0.0900666, 0.103923],
    ]
    boundary_ratio = 0.5  # in bc : in inner
    b_name = np.array(["left", "right", "bottom"])
    original_indices = np.arange(len(b_name))
    shuffled_indices = np.random.permutation(original_indices)
    shuffle_b = b_name[shuffled_indices]
    if shuffle_b[0] == "bottom":
        coord_x = coord_x_all[0]
    elif shuffle_b[0] == "left":
        coord_x = coord_x_all[1]
    elif shuffle_b[0] == "right":
        coord_x = coord_x_all[2]

    coord_x_str = str(coord_x[0])
    disp_x_str = str(disp_x_b[0])
    disp_y_str = str(disp_y_b[0])
    for i in range(1, len(coord_x)):
        coord_x_str += " " + str(coord_x[i])
        disp_x_str += " " + str(disp_x_b[i])
        disp_y_str += " " + str(disp_y_b[i])

    rand_n = random.random()

    disp_x_ic = np.mean(disp_x_b)
    disp_y_ic = np.mean(disp_y_b)
    free_emb = np.array([0, 1, 0])  # np.ones((1, 8)) * -1
    sys_emb = np.array([0, 0, 1])  # np.ones((1, 8)) * -2

    def nomalize(x):
        return (x + 4e-4) / 1.6e-3 * 2 - 1  # normalize to (-1,1)

    dir_emb = np.array(
        [1, nomalize(disp_xl), nomalize(disp_yl)]
    )  # np.concatenate((disp_x_b, disp_y_b), axis=0).reshape(1, -1)
    # F:free,f:fix,s:symmetry;FFf,ssF,ssf,Ffs
    if rand_n < 1 / 4:  # fFF
        b1 = dir_b % {"b": shuffle_b[0]}
        b2 = Free
        b3 = Free
        boundary_condition = np.vstack((dir_emb, free_emb, free_emb))
        fun = function
    elif 1 / 4 < rand_n < 2 / 4:  # ssF
        b1 = sys_b_sub % {"b": shuffle_b[0]}
        b2 = sys_b_sub % {"b": shuffle_b[1]}
        b1 = sys_b % {"module": b1 + b2}
        b2 = Free
        b3 = Free
        boundary_condition = np.vstack((sys_emb, sys_emb, free_emb))
        fun = Free
        disp_x_ic = 0
        disp_y_ic = 0
    elif 2 / 4 < rand_n < 3 / 4:  # fss
        b1 = dir_b % {"b": shuffle_b[0]}
        b2 = sys_b_sub % {"b": shuffle_b[1]}
        b3 = sys_b_sub % {"b": shuffle_b[2]}
        b2 = sys_b % {"module": b2 + b3}
        b3 = Free
        boundary_condition = np.vstack((dir_emb, sys_emb, sys_emb))
        fun = function
    else:  # fsF
        b1 = dir_b % {"b": shuffle_b[0]}
        b2 = sys_b % {"module": sys_b_sub % {"b": shuffle_b[1]}}
        b3 = Free
        boundary_condition = np.vstack((dir_emb, sys_emb, free_emb))
        fun = function
    boundary_condition = boundary_condition[np.argsort(shuffled_indices)]
    write_inp("./disp_base.txt", "./disp_x.txt", replacements={"x_coor": coord_x_str, "data": disp_x_str})
    write_inp("./disp_base.txt", "./disp_y.txt", replacements={"x_coor": coord_x_str, "data": disp_y_str})
    replacements = {"bc_1": b1, "bc_2": b2, "bc_3": b3, "fix_b": fun, "disp_x_ic": disp_x_ic, "disp_y_ic": disp_y_ic}
    return replacements, boundary_condition


def gen_boundary_without_fix(*arg):
    b_name = np.array(["left", "right", "bottom"])
    original_indices = np.arange(len(b_name))
    shuffled_indices = np.random.permutation(original_indices)
    shuffle_b = b_name[shuffled_indices]

    free_emb = np.array([0, 0, 0])  # np.ones((1, 8)) * -1
    sys_emb = np.array([0, 1, 1])  # np.ones((1, 8)) * -2

    b1 = sys_b_sub % {"b": shuffle_b[0]}
    b2 = sys_b_sub % {"b": shuffle_b[1]}
    b1 = sys_b % {"module": b1 + b2}
    b2 = Free
    b3 = Free
    boundary_condition = np.vstack((sys_emb, sys_emb, free_emb))
    fun = Free
    boundary_condition = boundary_condition[np.argsort(shuffled_indices)]
    disp_x_ic = 0
    disp_y_ic = 0
    replacements = {"bc_1": b1, "bc_2": b2, "bc_3": b3, "fix_b": fun, "disp_x_ic": disp_x_ic, "disp_y_ic": disp_y_ic}
    return replacements, boundary_condition


def write_val(n_block=64, up=8e5, down=2e5):
    replacement = ""
    flux = np.random.uniform(low=down, high=up, size=n_block)
    for i in range(n_block):
        replacement += ic_flux % {"block": 101 + i, "flux": flux[i]}
    replacements = {"ic_flux": replacement}
    write_inp("./val_base.i", "./val.i", replacements)
    np.save("./val_flux.npy", flux)
    return


def main(n, mpi=4):
    flux = []
    boundary = []
    num_data = 0
    for i in tqdm(range(n), desc="calculate loop time step", total=n):
        l_b, u_b = 0.1, 1
        # l_b, u_b = 760 * 1e-6, 850 * 1e-6
        Z = gen_flux1(l_b, u_b)

        b_replacements, boundary_condition = gen_boundary_without_fix("./represent_base.i", "./represent.i")

        # flux.append(Z)
        # boundary.append(boundary_condition)

        replacements = replacement_inp(Z)
        replacements.update(b_replacements)
        write_inp("./represent_base.i", "./represent.i", replacements)

        # command = "mpiexec " + "-n " + str(mpi) + " ../workspace-opt " + "-i " + "represent.i"
        # os.system(command)
        command = ["mpiexec", "-n", str(mpi), "../workspace-opt", "-i", "represent.i"]
        result = subprocess.run(command, capture_output=True, text=True)
        if "Solve Did NOT Converge" in result.stdout:
            print("error in ", i)
            continue
        else:
            num_data += 1
            flux.append(Z)
            boundary.append(boundary_condition)
            outfile = "./output/represent_out" + "_" + str(num_data) + ".e"
            command = "mv" + " ./represent_exodus.e " + outfile
            os.system(command)
    print("success case", num_data)
    np.save("./output/flux", np.array(flux))
    np.save("./output/bc", np.array(boundary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data")
    parser.add_argument("--type", default="val", type=str, help="train or val")
    parser.add_argument("--n", default="100", type=int, help="number of sample")
    # parser.add_argument("--index", default="1", type=int, help="start index")
    parser.add_argument("--mpi", default="4", type=int, help="mpiexec")
    args = parser.parse_args()
    if args.type == "train":
        main(args.n, args.mpi)
    else:
        write_val()
