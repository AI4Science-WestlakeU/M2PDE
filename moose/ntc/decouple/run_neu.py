import os
import shutil
import numpy as np
from scipy.interpolate import make_interp_spline
import subprocess
import netCDF4 as nc
from tqdm.auto import tqdm
import argparse

relaxation = 0.5
try:
    read_from_other_field = True
    T_fuel = np.load("./output/nft_Tfuel.npy")[:, 0, 1:].transpose(0, 2, 3, 1)
    T_fluid = np.load("./output/nft_Tfluid.npy")[:, 0, 1:].transpose(0, 2, 3, 1)
    try:
        T_fuel_old = np.load("./output/nft_Tfuel_old.npy")[:, 0, 1:].transpose(0, 2, 3, 1)
        T_fluid_old = np.load("./output/nft_Tfluid_old.npy")[:, 0, 1:].transpose(0, 2, 3, 1)
    except:
        T_fuel_old = T_fuel
        T_fluid_old = T_fluid
except:
    read_from_other_field = False
    print("using constant temperature")
try:
    phi_BC = np.load("./output/phiBC_to_phi.npy")
except:
    print("using new phi_BC")


def write_inp(base_file, out_file, replacements):
    #'./inpbase.txt' './neutroninp/phi.txt fluidT.txt fuelT.txt'
    with open(base_file, "r", encoding="utf-8") as base:
        template = base.read()
    src = template % replacements
    with open(out_file, "w", encoding="utf-8") as file:
        file.write(src)


def replacements(
    function, batch, tag="T", Lx=0.0076, Ly=0.75, Lt=5, nx=8, ny=64, nt=16, bias_x=0, bias_y=0, bias_t=0, dim=2
):
    # phi: Lx=0.0076, Ly=0.75, Lt=5, nx=8, ny=64, nt=16
    # Tfuel: Lx=0.0076, Ly=0.75, Lt=5, nx=8, ny=64, nt=16
    # Tfluid: Lx=0.0114, Ly=0.75, Lt=5, nx=12, ny=64, nt=16, bias_y = 0.0075
    dx = Lx / nx
    dy = Ly / ny
    dt = Lt / nt
    coor_t_str = ""
    data = ""
    if read_from_other_field == False or tag == "phi":  # fuel is difine in mesh point
        coor_x_str = "%.5f" % bias_x
        coor_y_str = "%.5f" % bias_y
        for i in range(nx):
            coor_x_str = coor_x_str + " %.5f" % (dx * (i + 1) + bias_x)
        for i in range(ny):
            coor_y_str = coor_y_str + " %.5f" % (dy * (i + 1) + bias_y)
    else:  # fluid is define in mesh
        coor_x_str = ""
        coor_y_str = ""
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
    if dim == 2:
        Z, inputs = function(batch, X, Y, T)
        for i in range(nt):
            for j in range(len(y_values)):
                for k in range(len(x_values)):
                    data += "%.2f " % Z[k, j, i]
        replacements = {"x_coor": coor_x_str, "y_coor": coor_y_str, "t_coor": coor_t_str, "data": data}
    else:
        Z = function(batch)
        for i in range(nt):
            for j in range(len(y_values)):
                data += "%.2f " % Z[j, i]
        replacements = {"y_coor": coor_y_str, "t_coor": coor_t_str, "data": data}
        inputs = Z
    return replacements, inputs


def gen_phi_BC(batch, *arg):
    try:
        return phi_BC[batch]
    except:

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
            y_random[max_index] += (1 - factor[max_index]) * peak_increase  #
            y_all = np.concatenate(([y_fixed[0]], y_random, [y_fixed[1]]))
            spline = make_interp_spline(x_all, y_all)
            y_spline = np.abs(spline(x_spline))
            phi_all.append(np.abs(y_spline))
        phi_all = np.array(phi_all).transpose(1, 0)
        return phi_all


def gen_T_fuel(batch, x, *arg):
    try:
        return T_fuel[batch] * (relaxation) + (1 - relaxation) * T_fuel_old[batch]
    except:
        cos_z = 400 * np.sin(np.linspace(0, 3.14, 80))[:65].reshape(1, 65, 1)
        # return np.ones_like(x) * 560
        return np.ones_like(x) * 560 + cos_z


def gen_T_fluid(batch, x, *arg):
    try:
        return T_fluid[batch] * (relaxation) + (1 - relaxation) * T_fluid_old[batch]
    except:
        cos_z = 200 * np.sin(np.linspace(0, 3.14, 80))[:65].reshape(1, 65, 1)
        # return np.ones_like(x) * 560
        return np.ones_like(x) * 560 + cos_z


def gen_sigma_af_fluid(batch, x, y, z):
    T = gen_T_fluid(batch, x, y, z)
    return -(20 + 20 * (T - 560) / (800 - 560)), T


def gen_sigma_af_fuel(batch, x, y, z):
    T = gen_T_fuel(batch, x, y, z)
    return (
        2.416 * 583.5 * 1.305 * 1.602 * 0.1
        - (13.47 * (T - 560) / (900 - 560) + 7.53) * 2.1479 * 1.602
        + 0.185 * 6.6072 * 0.1
        - 680.9 * 1.305 * 1.602 * 0.1
    ), T


def gen_neu_inp(batch):
    replacements_phi, phi = replacements(
        function=gen_phi_BC,
        batch=batch,
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
    replacements_sigma_af_fluid, Tfluid = replacements(
        function=gen_sigma_af_fluid,
        batch=batch,
        Lx=0.0114,
        Ly=0.75,
        Lt=5,
        nx=12,
        ny=64,
        nt=16,
        bias_x=0.0076,
        bias_y=0,
        bias_t=0,
    )
    replacements_sigma_af_fuel, Tfuel = replacements(
        function=gen_sigma_af_fuel,
        batch=batch,
        Lx=0.0076,
        Ly=0.75,
        Lt=5,
        nx=8,
        ny=64,
        nt=16,
        bias_x=0,
        bias_y=0,
        bias_t=0,
    )
    write_inp("./inp1D_base.txt", "./neutroninp/phi.txt", replacements_phi)
    write_inp("./inp_base.txt", "./neutroninp/sigma_af_fuel.txt", replacements_sigma_af_fuel)
    write_inp("./inp_base.txt", "./neutroninp/sigma_af_fluid.txt", replacements_sigma_af_fluid)
    inp_file = ["'neutroninp/phi.txt'", "'neutroninp/sigma_af_fuel.txt'", "'neutroninp/sigma_af_fluid.txt'"]
    replacements_inp = {"phi_in": inp_file[0], "sigma_af_fuel": inp_file[1], "sigma_af_fluid": inp_file[2]}
    write_inp(base_file="./neutron_base.i", out_file="./neutron.i", replacements=replacements_inp)
    return phi, Tfuel, Tfluid


def read_e_to_np(file_path):
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
    assert np.mean(mask) == 1, "An element has not been assigned a value"
    return z_matrix


def rename(original_file_path, new_file_path):
    if os.path.exists(original_file_path):
        if os.path.exists(new_file_path):
            os.remove(new_file_path)

        #
        shutil.move(original_file_path, new_file_path)


def main(n):
    # this file should be run in current file path
    phiBC_all = []
    T_fuel_all = []
    T_fluid_all = []
    outputs_all = []
    for i in tqdm(range(n), desc="calculate loop time step", total=n):
        phiBC, Tfuel, Tfluid = gen_neu_inp(i)
        command = ["mpiexec", "-n", "1", "../../workspace-opt", "-i", "neutron.i"]
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stderr)
        outputs = read_e_to_np("./neutron_exodus.e")
        phiBC_all.append(phiBC)
        T_fuel_all.append(Tfuel)
        T_fluid_all.append(Tfluid)
        outputs_all.append(outputs)
    phiBC_all = np.array(phiBC_all)
    T_fuel_all = np.array(T_fuel_all)
    T_fluid_all = np.array(T_fluid_all)
    outputs_all = np.array(outputs_all)
    print("neu: ", outputs_all.shape)
    print("T_fuel: ", T_fuel_all.shape)
    print("T_fluid: ", T_fluid_all.shape)
    print("phiBC_all: ", phiBC_all.shape)
    rename("./output/phiBC_to_phi.npy", "./output/phiBC_to_phi_old.npy")
    np.save("./output/phiBC_to_phi", np.array(phiBC_all))
    rename("./output/Tfuel_to_phi.npy", "./output/Tfuel_to_phi_old.npy")
    np.save("./output/Tfuel_to_phi", np.array(T_fuel_all))
    rename("./output/Tfluid_to_phi.npy", "./output/Tfluid_to_phi_old.npy")
    np.save("./output/Tfluid_to_phi", np.array(T_fluid_all))
    rename("./output/nft_phi.npy", "./output/nft_phi_old.npy")
    np.save("./output/nft_phi", np.array(outputs_all))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data")
    parser.add_argument("--n", default="1", type=int, help="number of sample")
    args = parser.parse_args()
    main(args.n)
