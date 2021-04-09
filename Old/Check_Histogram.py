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
    n_bins = 100
    h_max  = 4e2
    h_min  = 0

    ## Loading the stats
    stat_file = inpt_folder + "stats.csv"
    stats = np.loadtxt(stat_file, skiprows=1, delimiter=",")
    means =  T.from_numpy(stats[0,-2:])
    devs  =  T.from_numpy(stats[1,-2:])

    i = 0
    for weight_to in [0, 2e5, 2.5e5]:

        ## Setting up the weights
        max_weight = 0
        if weight_to > 0:
            weight_file = inpt_folder + "weights.csv"
            weights = np.loadtxt(weight_file, delimiter=",")
            max_weight = weights[ (np.abs(weights[:,0] - weight_to)).argmin(), 1 ]

        ## Loading the dataset
        inpt_files = [ f for f in glob.glob( inpt_folder + "*.h5" ) ]

        # dataset = myDS.StreamMETDataset( inpt_files, 5, 4096, max_weight )
        dataset = myDS.METDataset( inpt_files, max_weight )
        loader = DataLoader( dataset, batch_size=4096, drop_last=False, num_workers=4, sampler=dataset.sampler )

        ## Initialising the hitogram
        bins = np.linspace( h_min, h_max, n_bins+1 ) ## The edges of the histogram
        bw = bins[1] - bins[0]
        hist = np.zeros(n_bins)

        loader.dataset.weight_off()
        
        for (inputs, targets) in tqdm(loader, ncols=80, unit=""):

            real_targ = ( targets*devs + means ) / 1000
            targ_mag = T.linalg.norm(real_targ, dim=1)
            bin = np.clip( ( targ_mag / bw ).int().numpy(), 0, n_bins-1 )
            idx, cnts = np.unique(bin, return_counts=True)
            hist[idx] += cnts
            break

        c = cmap( (i+0.5) / 3 )
        plt.step( bins, np.insert(hist,0,0), color=c )
        i += 1
    plt.show()

if __name__ == "__main__":
    main()
