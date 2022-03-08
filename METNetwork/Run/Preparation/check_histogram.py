import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

import torch as T
from torch.utils.data import DataLoader

from METNetwork.resources.datasets import StreamMETDataset
from METNetwork.Resources import Plotting as myPL

T.set_grad_enabled(False)
cmap = plt.get_cmap("turbo")


def main():

    ## Create the dict for the dataset confid
    data_conf = {
        "dset": "val",
        "path": "/mnt/scratch/Data/METData/HDFs/WithSigAndEtaFix/",
        "do_rot": False,
        "inpts_rmv": ",",  ## Should remove all inputs except 1
        "n_ofiles": 64,
        "chunk_size": 3990,
        "sampler_kwargs": {},
    }

    loader_conf = {
        "batch_size": 4096,
        "num_workers": 15,
        "drop_last": False,
        "pin_memory": True,
    }

    ## The bin structure of the histograms
    n_bins = 101
    mag_bins = np.linspace(0, 450, n_bins + 1)
    trg_bins = [
        np.linspace(-4, 4, n_bins + 1) + data_conf["do_rot"],
        np.linspace(-4, 4, n_bins + 1),
    ]
    exy_bins = [
        np.linspace(-400, 400, n_bins + 1) + 100 * data_conf["do_rot"],
        np.linspace(-400, 400, n_bins + 1),
    ]

    ## Choosing the parameters weighting tests
    weight_type = ["mag"]
    weight_from = [0.4]
    weight_to = [3.5]
    weight_shift = [0]
    weight_ratio = [0]

    ## Make all possible test options in a grid
    tests = np.array(
        np.meshgrid(weight_type, weight_from, weight_to, weight_shift, weight_ratio)
    ).T.reshape(-1, 5)
    tests = np.concatenate([[[0, 0, 0, 0, 0]], tests])  ## Concatenate truth, untouched!

    ## The list of all the histograms to be filled
    mag_list = []
    trg_list = []
    exy_list = []

    ## Cycle through all possible tests
    for i, (w_type, w_from, w_to, w_shift, w_ratio) in enumerate(tests):

        ## Add the sampler kwargs to the config
        data_conf["sampler_kwargs"] = {
            "weight_type": w_type,
            "weight_from": float(w_from),
            "weight_to": float(w_to),
            "weight_shift": float(w_shift),
            "weight_ratio": float(w_ratio),
        }

        ## Create the data class
        dataset = StreamMETDataset(**data_conf)
        loader = DataLoader(dataset, **loader_conf)

        ## Get the stats
        stats = dataset.get_preprocess_info()
        outp_means = stats["outp_means"]
        outp_sdevs = stats["outp_sdevs"]

        ## Initialise the histograms histograms
        mag_hist = np.zeros(n_bins)
        trg_hist = np.zeros((n_bins, n_bins))
        exy_hist = np.zeros((n_bins, n_bins))

        ## Cycle through the dataset
        for (inputs, targets, weights) in tqdm(loader):

            ## Un-normalise the targets and convert to GeV
            tru_xy = (targets * outp_sdevs + outp_means) / 1000
            tru_et = T.norm(tru_xy, dim=1)

            ## Convert the required tensors back to numpy arrays
            weights = weights.numpy()
            targets = targets.numpy()
            tru_xy = tru_xy.numpy()
            tru_et = tru_et.numpy()

            ## Update the histograms
            mag_hist += np.histogram(tru_et, mag_bins, weights=weights)[0]
            trg_hist += np.histogram2d(*targets.T, trg_bins, weights=weights)[0]
            exy_hist += np.histogram2d(*tru_xy.T, exy_bins, weights=weights)[0]

        ## Normalise
        mag_hist /= np.sum(mag_hist)
        trg_hist /= np.sum(trg_hist)
        exy_hist /= np.sum(exy_hist)

        ## Add the histogram to their list
        mag_list.append(mag_hist)
        trg_list.append(trg_hist)
        exy_list.append(exy_hist)

    names = ["Truth"]
    names += [
        "w_type = {}, w_from = {}, w_to = {}, w_shift = {}, w_ratio = {}".format(*cnfg)
        for cnfg in tests[1:]
    ]

    ## Save the Magnitude histograms
    myPL.plot_and_save_hists(
        Path("S_MagDistTR"),
        mag_list,
        names,
        [r"$p_\mathrm{T}^\mathrm{miss}$ [GeV]", "Normalised"],
        mag_bins,
    )
    myPL.plot_and_save_contours(
        Path("S_TrgDistTR"),
        trg_list,
        names,
        ["scaled x", "scaled y"],
        trg_bins,
        do_csv=True,
    )
    myPL.plot_and_save_contours(
        Path("S_ExyDistTR"),
        exy_list,
        names,
        [r"$p_{x}^\mathrm{miss}$ [GeV]", r"$p_{y}^\mathrm{miss}$ [GeV]"],
        exy_bins,
    )


if __name__ == "__main__":
    main()
