import netCDF4 as nc
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# import path
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.filepath import ABSOLUTE_PATH

sys.path.append(ABSOLUTE_PATH)


def unique_within_tolerance(arr, tol):
    sorted_arr = np.sort(arr)
    unique = [sorted_arr[0]]
    for i in range(1, len(sorted_arr)):
        if np.abs(sorted_arr[i] - unique[-1]) > tol:
            unique.append(sorted_arr[i])
    return np.array(unique)


def read_e_to_np(file_path):
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
    print("the shape is: ", time_steps, len(unique_x), len(unique_y))
    z_matrix = np.zeros((time_steps, len(unique_x), len(unique_y)))
    mask = np.zeros((len(unique_x), len(unique_y)))
    for i in range(len(x_coords)):
        x_index = np.argmin(np.abs(x_coords[i] - unique_x))
        y_index = np.argmin(np.abs(y_coords[i] - unique_y))
        mask[x_index, y_index] = 1
        z_matrix[:, x_index, y_index] = u[:, i]
    assert np.mean(mask) == 1, "An element has not been assigned a value"
    return z_matrix


def read_e_file(file_path):
    dataset = nc.Dataset(file_path, mode="r")
    return dataset


def extract_block_data(dataset, block_id):
    global data_type
    element_connectivity = dataset.variables[f"connect{block_id+1}"][:]

    x_coords = dataset.variables["coordx"][:]
    y_coords = dataset.variables["coordy"][:]

    temperature = dataset.variables["vals_nod_var1"][-1]
    if data_type == "disp":
        para_x = dataset.variables["vals_nod_var2"][-1]
        para_y = dataset.variables["vals_nod_var3"][-1]
    elif data_type == "strain":
        para_x = dataset.variables["vals_nod_var5"][-1]
        para_y = dataset.variables["vals_nod_var6"][-1]
    unique_node_indices = np.unique(element_connectivity)

    block_x_coords = np.array(x_coords[unique_node_indices - 1])
    block_y_coords = np.array(y_coords[unique_node_indices - 1])
    block_temperature = np.array(temperature[unique_node_indices - 1])
    block_para_x = np.array(para_x[unique_node_indices - 1])
    block_para_y = np.array(para_y[unique_node_indices - 1])

    return unique_node_indices, block_x_coords, block_y_coords, block_temperature, block_para_x, block_para_y


def create_graph(unique_node_indices, block_x_coords, block_y_coords, block_temperature, block_disp_x, block_disp_y):
    G = nx.Graph()

    for i in range(len(unique_node_indices)):
        G.add_node(
            unique_node_indices[i],
            x=block_x_coords[i],
            y=block_y_coords[i],
            temperature=block_temperature[i],
            disp_x=block_disp_x[i],
            disp_y=block_disp_y[i],
        )

    return G


def get_internal_edges(element_connectivity):
    edges = set()
    for element in element_connectivity:
        edges.add((element[0], element[1]))
        edges.add((element[1], element[2]))
        edges.add((element[2], element[3]))
        edges.add((element[3], element[0]))
        edges.add((element[1], element[0]))
        edges.add((element[2], element[1]))
        edges.add((element[3], element[2]))
        edges.add((element[0], element[3]))
    return list(edges)


def draw_graph(graph, block_id):
    pos = {node: (data["x"], data["y"]) for node, data in graph.nodes(data=True)}
    temperatures = nx.get_node_attributes(graph, "temperature")
    node_colors = [temperatures[node] for node in graph.nodes()]
    plt.figure(figsize=(32, 32))
    nx.draw(graph, pos, node_color=node_colors, with_labels=False, node_size=50, edge_color="gray")
    plt.title(f"Block {block_id} Graph")
    plt.show()


def sym_block(x_coords, y_coords, sym_axis_x, sym_axis_y):
    if sym_axis_x > 0:
        x_coords = 2 * sym_axis_x - x_coords
    if sym_axis_y > 0:
        y_coords = 2 * sym_axis_y - y_coords
    return x_coords, y_coords


def move_block(x_coords, y_coords, move):
    return x_coords + move[0], y_coords + move[1]


def coord_transform(block_id, x, y, decimals=5):
    move_dis = -1 * np.array(
        [
            [0, 0],
            [-0.013856407, -0.024],
            [0.013856406, -0.024],
            [-0.027712807, -0.048],
            [0.0, -0.048],
            [0.027712813, -0.048],
            [-0.041569220, -0.072],
            [-0.013856407, -0.072],
            [0.013856406, -0.0720],
            [0.0415692190, -0.072],
            [0, 0],
            [-0.013856407, -0.024],
            [0.013856406, -0.024],
            [-0.027712807, -0.048],
            [0.0, -0.048],
            [0.027712813, -0.048],
        ]
    )
    sym_y = [-1] * 10 + [0.072] * 6
    sym_x = [-1] * 10 + [0.05542562] * 6
    if block_id > 1:
        x, y = move_block(x, y, move_dis[block_id - 1])
        x, y = sym_block(x.copy(), y.copy(), sym_x[block_id - 1], sym_y[block_id - 1])
    return np.round(x, decimals=decimals), np.round(y, decimals=decimals)


def draw_graph2(graph, block_id):
    x = []
    y = []
    z = []
    for node, data in graph.nodes(data=True):
        x.append(data["x"])
        y.append(data["y"])
        z.append(data["temperature"])
    triang = Triangulation(np.array(x), np.array(y))
    plt.tricontourf(triang, np.array(z), levels=20, cmap="viridis")
    plt.colorbar(label="Physical Quantity")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Contour Plot of Physical Quantity")
    plt.show()


def draw_graph_all(node_feature, num_nodes=804, num_block=6):
    x = []
    y = []
    z = []
    values = []
    adjs = []
    connectivity = np.load(ABSOLUTE_PATH + "/data/connect.npy")
    adj = np.array(get_internal_edges(connectivity)) - 804
    for i in range(num_block):
        # 804 nodes
        adj = adj + 804
        adjs = adjs + list(adj)
        for j in range(num_nodes):
            x.append(node_feature[i, j, 0])
            y.append(node_feature[i, j, 1])
            z.append(node_feature[i, j, 2])

    G = nx.Graph()

    for i in range(len(x)):
        G.add_node(
            i,
            x=x[i],
            y=y[i],
            phy=z[i],
        )
    G.add_edges_from(adjs)
    pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
    temperatures = nx.get_node_attributes(G, "phy")
    node_colors = [temperatures[node] for node in G.nodes()]
    plt.figure(figsize=(32, 32))
    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=50, edge_color="gray")
    plt.show()


def read_e_val(path):
    dataset = nc.Dataset(path, "r")

    all_phy = []
    coord = []
    for block_id in range(64):
        unique_node_indices, block_x_coords, block_y_coords, block_temperature, block_disp_x, block_disp_y = (
            extract_block_data(dataset, block_id)
        )
        block_temperature = (block_temperature - 948) / (1500 - 948) * 2 - 1
        # block_disp_x = (block_disp_x + 4e-4) / (1.6e-3) * 2 - 1
        # block_disp_y = (block_disp_y + 4e-4) / (1.6e-3) * 2 - 1
        block_disp_x = (block_disp_x - 0.00268) / (0.00685 - 0.00268) * 2 - 1
        block_disp_y = (block_disp_y - 0.00268) / (0.00685 - 0.00268) * 2 - 1
        all_phy.append(np.column_stack((block_temperature, block_disp_x, block_disp_y)))
        coord.append(np.column_stack((block_x_coords, block_y_coords)))
    all_phy = np.array(all_phy).reshape(-1, 3)
    coord = np.array(coord).reshape(-1, 2)
    np.save(ABSOLUTE_PATH + "/data/heatpipe/val_y.npy", all_phy)
    np.save(ABSOLUTE_PATH + "/data/heatpipe/coord_val.npy", coord)


def read_e(flux, bc, path, tolerance=1e-4, decimals=10):
    assert bc.shape == (3, 3), "bc shape is wrong"
    assert len(flux) == 16, "flux shape is wrong"
    flux = (flux - 1e5) / 9e5 * 2 - 1
    dataset = nc.Dataset(path, "r")

    all_phy = []
    for block_id in range(16):
        unique_node_indices, block_x_coords, block_y_coords, block_temperature, block_disp_x, block_disp_y = (
            extract_block_data(dataset, block_id)
        )
        block_temperature = (block_temperature - 948) / (1500 - 948) * 2 - 1
        # block_disp_x = (block_disp_x + 4e-4) / (1.6e-3) * 2 - 1
        # block_disp_y = (block_disp_y + 4e-4) / (1.6e-3) * 2 - 1
        block_disp_x = (block_disp_x - 0.00268) / (0.00685 - 0.00268) * 2 - 1
        block_disp_y = (block_disp_y - 0.00268) / (0.00685 - 0.00268) * 2 - 1
        phy = np.column_stack((block_temperature, block_disp_x, block_disp_y))
        if block_id == 0:
            x_base, y_base = block_x_coords, block_y_coords
            coords_new = np.stack((x_base, y_base), axis=-1)
            np.save(ABSOLUTE_PATH + "/data/heatpipe/coord.npy", coords_new)
            element_connectivity = np.array(dataset.variables[f"connect{block_id+1}"][:])
            # np.save(ABSOLUTE_PATH + "/data/mesh.npy", element_connectivity)
            # internal_edges = get_internal_edges(element_connectivity)
            # np.save(ABSOLUTE_PATH + "/data/adj.npy", internal_edges)
            phy_sorted = phy
        else:
            x_tran, y_tran = coord_transform(
                block_id=block_id + 1, decimals=decimals, x=block_x_coords, y=block_y_coords
            )

            # coords_prime = np.stack((x_tran, y_tran), axis=-1)

            # coord_to_phy = {tuple(coord): z for coord, z in zip(coords_prime, phy)}

            phy_sorted = np.empty_like(phy)

            for i, (x_val, y_val) in enumerate(zip(x_base, y_base)):
                distances = np.sqrt((x_tran - x_val) ** 2 + (y_tran - y_val) ** 2)

                min_distance_index = np.argmin(distances)
                phy_sorted[i] = phy[min_distance_index]
                if distances[min_distance_index] > tolerance:
                    print(
                        f"Multiple matching coordinates found for ({block_id},{x_val}, {y_val}) within tolerance, the min distance is {distances[min_distance_index]}."
                    )
        all_phy.append(phy_sorted)
    neighbors = {
        1: ("left", "right", 11),
        2: ("left", 11, 12),
        3: (11, "right", 13),
        4: ("left", 12, 14),
        5: (12, 13, 15),
        6: (13, "right", 16),
        7: ("left", 14, "bottom"),
        8: (14, 15, "bottom"),
        9: (15, 16, "bottom"),
        10: (16, "right", "bottom"),
        11: (3, 2, 1),
        12: (5, 4, 2),
        13: (6, 5, 3),
        14: (8, 7, 4),
        15: (9, 8, 5),
        16: (10, 9, 6),
    }
    bc_dict = {"left": 0, "right": 1, "bottom": 2}
    # 1,0,0:sym;0,1,0;free;disp_x,disp_y,1:fix
    node_feature_all = []
    y = []
    for i in range(1, len(all_phy) + 1):
        neighbor_i = neighbors[i]
        node_feature = np.zeros((len(unique_node_indices), 10))
        for j, neighbor in enumerate(neighbor_i):
            if isinstance(neighbor, str):
                b_emb = bc[bc_dict[neighbor]]
                virtual_graph = np.tile(b_emb, (len(unique_node_indices), 1))
                node_feature[:, j * 3 : j * 3 + 3] = virtual_graph
            else:
                node_feature[:, j * 3 : j * 3 + 3] = all_phy[neighbor - 1]
        node_feature[:, -1:] = flux[i - 1]
        node_feature_all.append(node_feature)
        y.append(all_phy[i - 1])
    return node_feature_all, y


def read_folder(num_folder):
    x_all = []
    y_all = []
    for i in range(1, 1 + num_folder):
        path = ABSOLUTE_PATH + "/data/heatpipe/" + str(i)
        bc = np.load(path + "/bc.npy")
        flux = np.load(path + "/flux.npy")
        num_e = bc.shape[0]
        for j in range(num_e):
            x, y = read_e(flux[j], bc[j], path + "/represent_out_" + str(j + 1) + ".e", tolerance=5e-5, decimals=10)
            x_all = x_all + x
            y_all = y_all + y
        print("folder ", str(i), " is finished")
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    np.save(ABSOLUTE_PATH + "/data/heatpipe/x", x_all)
    np.save(ABSOLUTE_PATH + "/data/heatpipe/y", y_all)


def main2(start=0, end=16):
    file_path = ABSOLUTE_PATH + "/data/represent_out_1.e"
    dataset = nc.Dataset(file_path, "r")

    all_graphs = []
    all_graphs_adj = []
    for block_id in range(start, end):
        unique_node_indices, block_x_coords, block_y_coords, block_temperature, block_disp_x, block_disp_y = (
            extract_block_data(dataset, block_id)
        )
        node_feature = np.column_stack((block_x_coords, block_y_coords, block_temperature, block_disp_x, block_disp_y))
        all_graphs.append(node_feature)
        element_connectivity = np.array(dataset.variables[f"connect{block_id+1}"][:]) - 1
        # np.save(ABSOLUTE_PATH + "/data/connect.npy", element_connectivity)
        internal_edges = get_internal_edges(element_connectivity)
        all_graphs_adj.append(internal_edges)
        # draw_graph2(graph, block_id)
        # print(graph.nodes(data=True))
        # print(graph.edges())
    all_graphs = np.array(all_graphs)
    all_graphs_adj = np.array(all_graphs_adj)
    return all_graphs, all_graphs_adj


if __name__ == "__main__":
    data_type = "strain"
    read_e_val(ABSOLUTE_PATH + "/data/heatpipe/val_exodus.e")
    read_folder(2)
    # main()
    # node_feature, _ = main2(10)
    # draw_graph_all(node_feature)
