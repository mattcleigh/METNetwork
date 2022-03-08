import numpy as np
import pandas as pd
import time
from glob import glob
from tqdm import tqdm
from pathlib import Path

import torch as T

from METNetwork.Resources.Networks import METNetwork
from METNetwork.Resources.Utils import setup_input_list

from mattstools.torch_utils import sel_device, to_np, move_dev

T.manual_seed(42)


def main():

    ## To run over the GPU and the batch size for evaluation
    b_size = 10000
    device = sel_device("gpu")

    ## The main directory to load the data from
    data_folder = "/mnt/scratch/Data/METData/Test/*susy*"

    ## A list of the network names and the name of the csv file to add to the datafiles
    network_folder = "/mnt/scratch/Saved_Networks/METNet/"
    file_name = "models/net_latest"
    network_names = [
        ("WithFix/56055011_0_07_03_22", "WithFix"),
    ]

    ## Set pytorch to not track gradients
    T.set_grad_enabled(False)

    ## Load the list of input data files
    all_files = glob(data_folder + "*/*.train-sample.csv")

    if not all_files:
        raise ValueError("No input files found")

    ## Ignore ttbar and Zmumu for now (too large)
    # all_files = list(filter(lambda s: "Zmumu" in s, all_files))
    # all_files = list(filter(lambda s: "00.t" in s, all_files))

    ## Cycle through the requested networks
    for network_file, network_name in network_names:
        print(network_name)

        ## Load the network and configure for full pass evaluation
        inpt_list = setup_input_list("Final,_ET", True)

        net = METNetwork(
            inpt_list,
            True,
            n_out=2,
            depth=5,
            width=512,
            act_h="silu",
            nrm=True,
            drp=0,
        )
        net.load_state_dict(T.load(Path(network_folder, network_file, file_name)))
        net = net.to(device)
        net.do_proc = True
        net.eval()

        ## Cycle through the input all input files
        for file in tqdm(all_files):

            ## Register the buffers
            net_et = []
            net_ex = []
            net_ey = []

            ## Also save True ET to verify that output files are matched
            ## Goes through the same steps as the others
            tru_et = []

            ## Iterate through the buffers using pandas chunk
            batch_reader = pd.read_csv(file, chunksize=b_size, dtype=np.float32)
            for batch in tqdm(batch_reader, leave=False):

                ## Load the batch inputs, convert to torch and pass through net
                batch = T.from_numpy(batch.to_numpy())

                ## Ignore final four values: True_ET, EX, EY and DSID
                net_out = net(move_dev(batch[:, :-4], device))

                ## Add the outputs to the buffers
                net_et.append(T.linalg.norm(net_out, axis=1))
                net_ex.append(net_out[:, 0])
                net_ey.append(net_out[:, 1])
                tru_et.append(batch[:, -4])

            ## Combine the buffers of the entire file
            net_et = to_np(T.cat(net_et, axis=0))
            net_ex = to_np(T.cat(net_ex, axis=0))
            net_ey = to_np(T.cat(net_ey, axis=0))
            tru_et = to_np(T.cat(tru_et, axis=0))

            ## Convert to a dataframe
            net_df = pd.DataFrame(
                {
                    network_name + "_ET": net_et,
                    network_name + "_EX": net_ex,
                    network_name + "_EY": net_ey,
                    network_name + "_ETtru": tru_et,
                }
            )

            ## Save the dataframe
            file_path = Path(file)
            output_path = Path(
                file_path.parent, str(file_path.stem) + "_" + network_name + ".csv"
            )
            net_df.to_csv(output_path, index=False)
            print(output_path)


if __name__ == "__main__":
    main()
