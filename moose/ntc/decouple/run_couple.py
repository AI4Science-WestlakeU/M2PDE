import argparse
import subprocess
import numpy as np
from tqdm.auto import tqdm


def L2_norm(array):
    b, c = array.shape[0], array.shape[1]
    array = array.reshape(b, c, -1)
    norm = np.sum(np.sum(array**2, axis=2) ** 0.5)
    return norm / b / c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="outer couple")
    parser.add_argument("--n", default="20", type=int, help="number of outer couple")
    args = parser.parse_args()
    for i in tqdm(range(args.n), desc="calculate loop time step", total=args.n):
        command = ["python", "run_neu.py"]
        result = subprocess.run(command, capture_output=True, text=True)
        #
        print(result.stdout)  #
        print(result.stderr)  #
        command = ["python", "run_fuel.py"]
        result = subprocess.run(command, capture_output=True, text=True)
        #
        print(result.stdout)  #
        print(result.stderr)  #
        command = ["python", "run_fluid.py"]
        result = subprocess.run(command, capture_output=True, text=True)
        #
        print(result.stdout)  #
        print(result.stderr)  # （
        if i > 0:
            neu_old = neu
            solid_old = solid
            fluid_old = fluid
        neu = np.load("./output/nft_phi.npy")
        solid = np.load("./output/nft_Tfuel.npy")
        fluid = np.load("./output/nft_Tfluid.npy")
        if i > 0:
            relative_loss_neu = L2_norm(neu_old - neu) / L2_norm(neu)
            relative_loss_solid = L2_norm(solid_old - solid) / L2_norm(solid)
            relative_loss_fluid = L2_norm(fluid_old - fluid) / L2_norm(fluid)
            print(
                "loss in iter " + str(i) + " of neu, solid and fluid: ",
                relative_loss_neu,
                relative_loss_solid,
                relative_loss_fluid,
            )
            if all(num < 1e-3 for num in [relative_loss_neu, relative_loss_solid, relative_loss_fluid]):
                print("converse in", i)
                break
    print("up to max iteratiopn")
