import numpy as np
import pandas as pd
import time
from glob import glob
from tqdm import tqdm
from pathlib import Path

import torch as T

from mattstools.torch_utils import sel_device, to_np, move_dev

T.manual_seed(42)


def main():

    ## To run over the GPU and the batch size for evaluation
    device = "gpu"
    b_size = 10000

    ## The main directory to load the data from
    data_folder = "/mnt/scratch/Data/METData/Test/"

    ## A list of the network names and the name of the csv file to add to the datafiles
    network_folder = "/mnt/scratch/Saved_Networks/METNet"
    file_name = "models/net_best"
    network_names = [
        ("WithFix/56055011_0_07_03_22", "WithFixes"),
    ]

    ## Define the device to work on and set pytorch to not track gradients
    device = sel_device(device)
    T.set_grad_enabled(False)

    ## Load the list of input data files
    all_files = glob(data_folder + "*/*.train-sample.csv")

    if not all_files:
        raise ValueError("No input files found")

    ## Ignore ttbar and Zmumu for now (too large)
    all_files = [x for x in all_files if "Zmumu" not in x and "ttbar" not in x]
    all_files = [x for x in all_files if "0.train-sample.csv" in x]

    ## Cycle through the requested networks
    for network_file, network_name in network_names:
        print(network_name)

        ## Load the network and configure for full pass evaluation
        net = T.load(Path(network_folder, network_file, file_name), map_location=device)
        # net.device = device
        # net.do_proc = True
        # net.eval()

        ## Cycle through the input all input files
        for file in tqdm(all_files):

            ## Register the buffers
            net_et = []
            net_ex = []
            net_ey = []

            ## Iterate through the buffers using pandas chunk
            batch_reader = pd.read_csv(file, chunksize=b_size, dtype=np.float32)
            for batch in tqdm(batch_reader, leave=False):

                ## Load the batch inputs, convert to torch and pass through net
                ## Ignore final four values: True_ET, EX, EY and DSID
                net_out = net(move_dev(T.from_numpy(batch.to_numpy()[:, :-4]), device))

                ## Add the outputs to the buffers
                net_et.append(T.linalg.norm(net_out, axis=1))
                net_ex.append(net_out[:, 0])
                net_ey.append(net_out[:, 1])

            ## Combine the buffers of the entire file
            net_et = to_np(T.cat(net_et, axis=0))
            net_ex = to_np(T.cat(net_ex, axis=0))
            net_ey = to_np(T.cat(net_ey, axis=0))

            # ## Convert to a dataframe
            net_df = pd.DataFrame(
                {
                    network_name + "_ET": net_et,
                    network_name + "_EX": net_ex,
                    network_name + "_EY": net_ey,
                }
            )

            # ## Save the dataframe
            # file_path = Path(file)
            # output_path = Path(
            #     file_path.parent, str(file_path.stem) + "_" + network_name + ".csv"
            # )
            # net_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
