import sys
home_env = "."
sys.path.append(home_env)

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm

import torch as T
from torch.utils.data import DataLoader

from Resources import Datasets as myDS
cmap = plt.get_cmap("turbo")

def main():

    inpt_folder =  "../Data/Rotated/"
    stream = True
    h_max  = 4e2
    n_bins = 100
    weight_list = [ "i300" ]

    ## Loading the stats for un-normalisation
    stat_file = inpt_folder + "stats.csv"
    stats = np.loadtxt(stat_file, skiprows=1, delimiter=",")
    means =  T.from_numpy(stats[0,-2:])
    devs  =  T.from_numpy(stats[1,-2:])

    ## Loading the dataset
    inpt_files = [ f for f in glob.glob( inpt_folder + "*.h5" ) ]
    hist_file = inpt_folder + "hist.csv"

    for i, weight_type in enumerate(weight_list):

        dataset = myDS.StreamMETDataset( inpt_files, 32, 1024, hist_file, weight_type )
        loader = DataLoader( dataset, drop_last=False, batch_size=4096, sampler=dataset.sampler, num_workers=4, pin_memory=True )

        hist, bins = np.histogram([0], bins=n_bins, range=[0,h_max], weights=[0.1])

        for (inputs, targets, weights) in tqdm(loader, ncols=80, unit=""):

            real_targ = ( targets*devs + means ) / 1000
            targ_mag = T.linalg.norm(real_targ, dim=1)
            targ_mag = np.clip(targ_mag, None, h_max)
            hist += np.histogram( targ_mag, bins=n_bins, range=[0,h_max], weights=weights.numpy() )[0]
            # break

        hist = hist / np.sum(hist)
        c = cmap( (i+0.5) / len(weight_list) )
        plt.step( bins, np.insert(hist,0,0), color=c, label=weight_type )

        del dataset
        del loader


    plt.ylabel( "Counts" )
    plt.xlabel( "True MET [GeV]" )
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
