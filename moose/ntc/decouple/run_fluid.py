import os
import shutil
import numpy as np
from scipy.interpolate import make_interp_spline
import subprocess
import netCDF4 as nc
from tqdm.auto import tqdm
import argparse

relaxation = 0.5
flux = np.load("./output/nft_Tfuel.npy")[:, 1, 1:, -1]
try:
    flux_old = np.load("./output/nft_Tfuel_old.npy")[:, 1, 1:, -1]
except:
    flux_old = flux


def write_inp(base_file, out_file, replacements):
    #'./inpbase.txt' './neutroninp/phi.txt fluidT.txt fuelT.txt'
    with open(base_file, "r", encoding="utf-8") as base:
        template = base.read()
    src = template % replacements
    with open(out_file, "w", encoding="utf-8") as file:
        file.write(src)


def replacements(function, batch, Lx=0.0076, Ly=0.75, Lt=5, nx=8, ny=64, nt=16, bias_x=0, bias_y=0, bias_t=0, dim=2):
    # phi: Lx=0.0076, Ly=0.75, Lt=5, nx=8, ny=64, nt=16
    # Tfuel: Lx=0.0076, Ly=0.75, Lt=5, nx=8, ny=64, nt=16
    # Tfluid: Lx=0.0114, Ly=0.75, Lt=5, nx=12, ny=64, nt=16, bias_y = 0.0075
    dx = Lx / nx
    dy = Ly / ny
    dt = Lt / nt
    coor_x_str = ""
    coor_y_str = ""
    coor_t_str = ""
    data = ""
    for i in range(nx):
        coor_x_str = coor_x_str + " %.5f" % (dx * i + dx / 2 + bias_x)
    for i in range(ny):
        coor_y_str = coor_y_str + " %.5f" % (dy * i + dy / 2 + bias_y)
    for i in range(nt):
        coor_t_str = coor_t_str + "%.5f " % (dt * (i + 1) + bias_t)

    x_values = np.array(list(map(float, coor_x_str.split())))
    y_values = np.array(list(map(float, coor_y_str.split())))
    t_values = np.array(list(map(float, coor_t_str.split())))

    X, Y, T = np.meshgrid(x_values, y_values, t_values, indexing="ij")
    Z = function(batch)
    for i in range(nt):
        for j in range(len(y_values)):
            data += "%.1f " % Z[j, i]
    replacements = {"y_coor": coor_y_str, "t_coor": coor_t_str, "data": data}
    inputs = Z
    return replacements, inputs


def gen_flux(batch, *arg):
    return (flux[batch] * (relaxation) + (1 - relaxation) * flux_old[batch]).transpose(1, 0)


def gen_neu_inp(batch):
    replacements_flux, flux = replacements(
        function=gen_flux,
        batch=batch,
        Lx=0.0114,
        Ly=0.75,
        Lt=5,
        nx=12,
        ny=64,
        nt=16,
        bias_x=0,
        bias_y=0,
        bias_t=0,
        dim=1,
    )
    # replacements_D_fluid = replacements(function=gen_D_fluid, Lx=0.0076, Ly=0.75, Lt=5, nx=12, ny=64, nt=16, bias_x = 0, bias_y = 0, bias_t = 0)
    write_inp("./inp1D_base.txt", "./fluidinp/flux.txt", replacements_flux)
    inp_file = ["'./fluidinp/flux.txt'"]
    replacements_inp = {"flux_file": inp_file[0]}
    write_inp(base_file="./fluid_base.i", out_file="./fluid.i", replacements=replacements_inp)
    return flux


def read_e_to_np(file_path):
    def unique_within_tolerance(arr, tol):
        #
        sorted_arr = np.sort(arr)
        #
        unique = [sorted_arr[0]]
        for i in range(1, len(sorted_arr)):
            if np.abs(sorted_arr[i] - unique[-1]) > tol:
                unique.append(sorted_arr[i])
        # NumPy
        return np.array(unique)

    dataset = nc.Dataset(file_path, "r")

    #
    x_coords = dataset.variables["coordx"][:]
    y_coords = dataset.variables["coordy"][:]
    if "coordz" in dataset.variables:
        z_coords = dataset.variables["coordz"][:]
    else:
        z_coords = [0.0] * len(x_coords)

    #
    num_nodes = len(x_coords)

    #
    connectivity = dataset.variables["connect1"][:]
    blocks = dataset.variables["eb_names"][:]
    num_blocks = len(blocks)

    #
    time_steps = dataset.variables["time_whole"][:]
    num_time_steps = len(time_steps)
    T_fluid = np.array(dataset.variables["vals_elem_var1eb1"]).reshape(1, num_time_steps, 64, 12)
    # print(np.min(T_fluid), np.max(T_fluid))
    pressure = np.array(dataset.variables["vals_elem_var3eb1"]).reshape(1, num_time_steps, 64, 12)
    vel_x = np.array(dataset.variables["vals_elem_var4eb1"]).reshape(1, num_time_steps, 64, 12)
    vel_y = np.array(dataset.variables["vals_elem_var5eb1"]).reshape(1, num_time_steps, 64, 12)

    unique_x = unique_within_tolerance(np.array(x_coords), 1e-6)
    unique_y = unique_within_tolerance(np.array(y_coords), 1e-3)

    # print("the shape is: ", num_time_steps, len(unique_x), len(unique_y))
    z_matrix = np.concatenate((T_fluid, pressure, vel_x, vel_y), axis=0).transpose(0, 1, 3, 2)
    return z_matrix


def rename(original_file_path, new_file_path):
    if os.path.exists(original_file_path):
        if os.path.exists(new_file_path):
            os.remove(new_file_path)

        #
        shutil.move(original_file_path, new_file_path)


def main(n=2):
    # this file should be run in current file path
    flux_all = []
    outputs_all = []
    for i in tqdm(range(n), desc="calculate loop time step", total=n):
        flux = gen_neu_inp(i)
        #
        command = ["mpiexec", "-n", "2", "../../workspace-opt", "-i", "fluid.i"]
        #
        result = subprocess.run(command, capture_output=True, text=True)
        #
        # print(result.stdout)  #
        print(result.stderr)  #
        outputs = read_e_to_np("./fluid_exodus.e")
        flux_all.append(flux)
        outputs_all.append(outputs)
    flux_all = np.array(flux_all)
    outputs_all = np.array(outputs_all)
    print("flux: ", flux_all.shape)
    print("output: ", outputs_all.shape)
    rename("./output/flux_to_fluid.npy", "./output/flux_to_fluid_old.npy")
    np.save("./output/flux_to_fluid", np.array(flux_all))
    rename("./output/nft_Tfluid.npy", "./output/nft_Tfluid_old.npy")
    np.save("./output/nft_Tfluid", np.array(outputs_all))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data")
    parser.add_argument("--n", default="1", type=int, help="number of sample")
    args = parser.parse_args()
    main(args.n)
