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

def main():

    inpt_folder =  "../Data/Rotated/"
    weight_to = 2e5 ## The location of the falling edge of the plateau in MeV
    n_bins = 100
    h_max  = 3e2
    h_min  = 0

    ## Loading the stats
    stat_file = inpt_folder + "stats.csv"
    stats = np.loadtxt(stat_file, skiprows=1, delimiter=",")
    means =  T.from_numpy(stats[0,-2:])
    devs  =  T.from_numpy(stats[1,-2:])

    ## Setting up the weights
    max_weight = 0
    if weight_to > 0:
        weight_file = inpt_folder + "weights.csv"
        weights = np.loadtxt(weight_file, delimiter=",")
        max_weight = weights[ (np.abs(weights[:,0] - weight_to)).argmin(), 1 ]
    print(max_weight)

    ## Loading the dataset
    inpt_files = [ f for f in glob.glob( inpt_folder + "*.h5" ) ]
    dataset = myDS.METDataset( inpt_files, 5, 4096, max_weight )
    loader = DataLoader( dataset, batch_size=4096, drop_last=False, num_workers=4 )

    ## Initialising the hitogram
    bins = np.linspace( h_min, h_max, n_bins+1 ) ## The edges of the histogram
    bw = bins[1] - bins[0]
    hist = np.zeros(n_bins)

    for (inputs, targets) in tqdm(loader):

        real_targ = ( targets*devs + means ) / 1000
        targ_mag = T.linalg.norm(real_targ, dim=1)
        bin = np.clip( ( targ_mag / bw ).int().numpy(), 0, n_bins-1 )
        idx, cnts = np.unique(bin, return_counts=True)
        hist[idx] += cnts

    plt.step( bins, np.insert(hist,0,0) )
    plt.show()

if __name__ == "__main__":
    main()
