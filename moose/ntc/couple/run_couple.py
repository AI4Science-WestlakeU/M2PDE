import numpy as np
from scipy.interpolate import make_interp_spline
import subprocess
import netCDF4 as nc
from tqdm.auto import tqdm
import argparse


def read_neu_to_np(file_path):
    def unique_within_tolerance(arr, tol):
        sorted_arr = np.sort(arr)
        unique = [sorted_arr[0]]
        for i in range(1, len(sorted_arr)):
            if np.abs(sorted_arr[i] - unique[-1]) > tol:
                unique.append(sorted_arr[i])
        return np.array(unique)

    dataset = nc.Dataset(file_path, "r")

    x_coords = dataset.variables["coordx"][:]
    y_coords = dataset.variables["coordy"][:]
    if "coordz" in dataset.variables:
        z_coords = dataset.variables["coordz"][:]
    else:
        z_coords = [0.0] * len(x_coords)

    num_nodes = len(x_coords)

    connectivity = dataset.variables["connect1"][:]
    blocks = dataset.variables["eb_names"][:]
    num_blocks = len(blocks)

    time_steps = dataset.variables["time_whole"][:]
    num_time_steps = len(time_steps)
    u = dataset.variables["vals_nod_var3"]

    unique_x = unique_within_tolerance(np.array(x_coords), 1e-6)
    unique_y = unique_within_tolerance(np.array(y_coords), 1e-3)

    time_steps = u.shape[0]
    # print("the shape is: ", time_steps, len(unique_x), len(unique_y))
    z_matrix = np.zeros((time_steps, len(unique_x), len(unique_y)))
    mask = np.zeros((len(unique_x), len(unique_y)))
    for i in range(len(x_coords)):
        x_index = np.argmin(np.abs(x_coords[i] - unique_x))
        y_index = np.argmin(np.abs(y_coords[i] - unique_y))
        mask[x_index, y_index] = 1
        z_matrix[:, x_index, y_index] = u[:, i]
    z_matrix = (z_matrix[..., :-1] + z_matrix[..., 1:]) / 2
    z_matrix = (z_matrix[:, :-1] + z_matrix[:, 1:]) / 2
    assert np.mean(mask) == 1, "An element has not been assigned a value"
    return z_matrix


def read_fuel_to_np(file_path):
    def unique_within_tolerance(arr, tol):
        sorted_arr = np.sort(arr)
        unique = [sorted_arr[0]]
        for i in range(1, len(sorted_arr)):
            if np.abs(sorted_arr[i] - unique[-1]) > tol:
                unique.append(sorted_arr[i])
        return np.array(unique)

    dataset = nc.Dataset(file_path, "r")

    x_coords = dataset.variables["coordx"][:]
    y_coords = dataset.variables["coordy"][:]
    if "coordz" in dataset.variables:
        z_coords = dataset.variables["coordz"][:]
    else:
        z_coords = [0.0] * len(x_coords)

    num_nodes = len(x_coords)


    connectivity = dataset.variables["connect1"][:]
    blocks = dataset.variables["eb_names"][:]
    num_blocks = len(blocks)


    time_steps = dataset.variables["time_whole"][:]
    num_time_steps = len(time_steps)
    u = dataset.variables["vals_nod_var1"]
    flux = np.array(dataset.variables["vals_elem_var1eb1"]).reshape(1, num_time_steps, 64, 8).transpose(0, 1, 3, 2)
    # flux1 = dataset.variables["vals_nod_var3"]

    unique_x = unique_within_tolerance(np.array(x_coords), 1e-6)
    unique_y = unique_within_tolerance(np.array(y_coords), 1e-3)

    time_steps = u.shape[0]
    # print("the shape is: ", time_steps, len(unique_x), len(unique_y))
    z_matrix = np.zeros((1, time_steps, len(unique_x), len(unique_y)))
    mask = np.zeros((len(unique_x), len(unique_y)))
    for i in range(len(x_coords)):
        x_index = np.argmin(np.abs(x_coords[i] - unique_x))
        y_index = np.argmin(np.abs(y_coords[i] - unique_y))
        mask[x_index, y_index] = 1
        z_matrix[0, :, x_index, y_index] = u[:, i]
        # z_matrix[1, :, x_index, y_index] = flux[:, i]
        # z_matrix[2, :, x_index, y_index] = flux1[:, i]
    z_matrix = (z_matrix[:, :, 1:, 1:] + z_matrix[:, :, :-1, :-1]) / 2
    assert np.mean(mask) == 1, "An element has not been assigned a value"
    return np.concatenate((z_matrix, flux), axis=0)


def read_fluid_to_np(file_path):
    def unique_within_tolerance(arr, tol):
        sorted_arr = np.sort(arr)
        unique = [sorted_arr[0]]
        for i in range(1, len(sorted_arr)):
            if np.abs(sorted_arr[i] - unique[-1]) > tol:
                unique.append(sorted_arr[i])
        return np.array(unique)

    dataset = nc.Dataset(file_path, "r")

    x_coords = dataset.variables["coordx"][:]
    y_coords = dataset.variables["coordy"][:]
    if "coordz" in dataset.variables:
        z_coords = dataset.variables["coordz"][:]
    else:
        z_coords = [0.0] * len(x_coords)

    num_nodes = len(x_coords)
    connectivity = dataset.variables["connect1"][:]
    blocks = dataset.variables["eb_names"][:]
    num_blocks = len(blocks)
    time_steps = dataset.variables["time_whole"][:]
    num_time_steps = len(time_steps)
    T_fluid = np.array(dataset.variables["vals_elem_var1eb1"]).reshape(1, num_time_steps, 64, 12)
    pressure = np.array(dataset.variables["vals_elem_var3eb1"]).reshape(1, num_time_steps, 64, 12)
    vel_x = np.array(dataset.variables["vals_elem_var4eb1"]).reshape(1, num_time_steps, 64, 12)
    vel_y = np.array(dataset.variables["vals_elem_var5eb1"]).reshape(1, num_time_steps, 64, 12)

    unique_x = unique_within_tolerance(np.array(x_coords), 1e-6)
    unique_y = unique_within_tolerance(np.array(y_coords), 1e-3)

    # print("the shape is: ", num_time_steps, len(unique_x), len(unique_y))
    z_matrix = np.concatenate((T_fluid, pressure, vel_x, vel_y), axis=0).transpose(0, 1, 3, 2)
    return z_matrix


def gen_phi_BC(*arg):

    def generate_x_coords(min_val, max_val, n_points, threshold):
        x_coords = []
        while len(x_coords) < n_points:
            x = np.random.uniform(min_val, max_val)
            if (
                (len(x_coords) == 0 or all(abs(x - xi) > threshold for xi in x_coords))
                and abs(x - min_val) > threshold
                and abs(x - max_val) > threshold
            ):
                x_coords.append(x)
        return sorted(x_coords)

    phi_all = []
    time_steps = 16
    while True:
        x_random = generate_x_coords(0, 0.75, 5, 0.075)
        y_random = np.random.uniform(0.5, 3, 5)
        max_index = np.argmax(y_random)
        x_fixed = np.array([0, 0.75])
        y_fixed = np.array([0.5, 0.5])
        x_all = np.concatenate(([x_fixed[0]], x_random, [x_fixed[1]]))
        y_all = np.concatenate(([y_fixed[0]], y_random, [y_fixed[1]]))
        spline = make_interp_spline(x_all, y_all)
        x_spline = np.linspace(0, 0.75, 65)
        y_spline = spline(x_spline)
        if np.all(y_spline >= 0):
            break
    phi_all.append(y_spline)

    max_increase = np.random.uniform(0.1, 0.9)
    for t in range(1, time_steps):
        peak_increase = np.random.uniform(0.1, max_increase)
        factor = np.random.uniform(0.1, 0.7, 5)
        y_random += factor * peak_increase
        y_random[max_index] += (1 - factor[max_index]) * peak_increase
        y_all = np.concatenate(([y_fixed[0]], y_random, [y_fixed[1]]))
        spline = make_interp_spline(x_all, y_all)
        y_spline = np.abs(spline(x_spline))
        phi_all.append(np.abs(y_spline))
    phi_all = np.array(phi_all).transpose(1, 0)
    return phi_all


def replacements(function, tag="T", Lx=0.0076, Ly=0.75, Lt=5, nx=8, ny=64, nt=16, bias_x=0, bias_y=0, bias_t=0, dim=2):
    # phi: Lx=0.0076, Ly=0.75, Lt=5, nx=8, ny=64, nt=16
    # Tfuel: Lx=0.0076, Ly=0.75, Lt=5, nx=8, ny=64, nt=16
    # Tfluid: Lx=0.0114, Ly=0.75, Lt=5, nx=12, ny=64, nt=16, bias_y = 0.0075
    dx = Lx / nx
    dy = Ly / ny
    dt = Lt / nt
    coor_t_str = ""
    data = ""
    coor_x_str = "%.5f" % bias_x
    coor_y_str = "%.5f" % bias_y
    for i in range(nx):
        coor_x_str = coor_x_str + " %.5f" % (dx * (i + 1) + bias_x)
    for i in range(ny):
        coor_y_str = coor_y_str + " %.5f" % (dy * (i + 1) + bias_y)
    for i in range(nt):
        coor_t_str = coor_t_str + "%.5f " % (dt * (i + 1) + bias_t)

    x_values = np.array(list(map(float, coor_x_str.split())))
    y_values = np.array(list(map(float, coor_y_str.split())))
    t_values = np.array(list(map(float, coor_t_str.split())))

    X, Y, T = np.meshgrid(x_values, y_values, t_values, indexing="ij")
    Z = function()
    for i in range(nt):
        for j in range(len(y_values)):
            data += "%.2f " % Z[j, i]
    replacements = {"y_coor": coor_y_str, "t_coor": coor_t_str, "data": data}
    inputs = Z
    return replacements, inputs


def write_inp(base_file, out_file, replacements):
    #'./inpbase.txt' './neutroninp/phi.txt fluidT.txt fuelT.txt'
    with open(base_file, "r", encoding="utf-8") as base:
        template = base.read()
    src = template % replacements
    with open(out_file, "w", encoding="utf-8") as file:
        file.write(src)


def main(n=2):
    # this file should be run in current file path
    phiBC_all = []
    T_fuel_all = []
    T_fluid_all = []
    neu_all = []
    for i in tqdm(range(n), desc="calculate loop time step", total=n):
        replacements_phi, phiBC = replacements(
            function=gen_phi_BC,
            tag="phi",
            Lx=0.0076,
            Ly=0.75,
            Lt=5,
            nx=8,
            ny=64,
            nt=16,
            bias_x=0,
            bias_y=0,
            bias_t=0,
            dim=1,
        )
        write_inp("./phi_base.txt", "./phi.txt", replacements_phi)
        command = ["mpiexec", "-n", "1", "../../workspace-opt", "-i", "solid.i"]
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stderr)
        Tfuel = read_fuel_to_np("./solid_exodus.e")
        fluid = read_fluid_to_np("./solid_out_sub_app0_sub_app0_exodus.e")
        neutron = read_neu_to_np("./solid_out_sub_app0_exodus.e")
        phiBC_all.append(phiBC)
        T_fuel_all.append(Tfuel)
        T_fluid_all.append(fluid)
        neu_all.append(neutron)
    np.save("./output/phiBC_to_phi", np.array(phiBC_all))
    np.save("./output/nft_Tfuel", np.array(T_fuel_all))
    np.save("./output/nft_Tfluid", np.array(T_fluid_all))
    np.save("./output/nft_phi", np.array(neu_all))


def main1():
    # this file should be run in current file path
    T_fuel_all = []
    T_fluid_all = []
    neu_all = []
    command = ["mpiexec", "-n", "1", "../../workspace-opt", "-i", "solid.i"]
    result = subprocess.run(command, capture_output=True, text=True)
    Tfuel = read_fuel_to_np("./solid_exodus.e")
    fluid = read_fluid_to_np("./solid_out_sub_app0_sub_app0_exodus.e")
    neutron = read_neu_to_np("./solid_out_sub_app0_exodus.e")
    T_fuel_all.append(Tfuel)
    T_fluid_all.append(fluid)
    neu_all.append(neutron)
    np.save("./output/nft_Tfuel", np.array(T_fuel_all))
    np.save("./output/nft_Tfluid", np.array(T_fluid_all))
    np.save("./output/nft_phi", np.array(neu_all))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data")
    parser.add_argument("--n", default="2000", type=int, help="number of sample")
    parser.add_argument("--type", default="single", type=str, help="single or batch")
    args = parser.parse_args()
    if args.type == "single":
        main1()
    else:
        main(args.n)
