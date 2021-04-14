import sys
home_env = "."
sys.path.append(home_env)

import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch as T
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Resources import Datasets as myDS

cmap = plt.get_cmap("turbo")

def main():

    inpt_folder =  "../Data/Rotated/"
    stream = True

    ## Defining the histogram
    n_bins = 20
    bin_max = 400

    ## Loading the dataset
    inpt_files = [ f for f in glob.glob( inpt_folder + "*.h5" ) ]

    ## Split the file list according to the test_frac
    test_frac = 0.1
    # np.random.seed(0)
    # np.random.shuffle(inpt_files)
    # n_test  = np.clip( int(round(len(inpt_files)*test_frac)), 1, len(inpt_files)-1 )
    # n_train = len(inpt_files) - n_test
    # train_files = file_list[:-n_test]
    # inpt_files  = inpt_files[-n_test:]

    dataset = myDS.StreamMETDataset( inpt_files, 32, 1024, 0 )
    loader = DataLoader( dataset, drop_last=False, batch_size=4096, sampler=dataset.sampler, num_workers=4, pin_memory=True )

    ## Loading the stats
    stat_file = inpt_folder + "stats.csv"
    stats = np.loadtxt(stat_file, skiprows=1, delimiter=",")
    means =  T.from_numpy(stats[0,:])
    devs  =  T.from_numpy(stats[1,:])

    ## The performance metrics to be calculated are stored in a running total matrix
    met_names  = [ "Loss", "Res", "Mag", "Ang", "DLin" ]
    run_totals = np.zeros( ( 1+n_bins, 1+len(met_names) ) ) ## rows = (total, bin1, bin2...) x cols = (n_events, *met_names)

    for (inputs, targets) in tqdm( loader, desc="Performance", ncols=80, unit="" ):

        ## Un-normalise the targets
        real_targ = ( targets*devs[-2:] + means[-2:] ) / 1000
        targ_mag = T.norm(real_targ, dim=1)

        ## Get the tight met variables
        zeros = T.zeros( [len(inputs), 1] )
        outputs = T.hstack( [inputs[:,:1], zeros] )
        real_outp = ( inputs[:,:1]*devs[0] + means[0] ) / 1000
        real_outp = T.hstack( [real_outp, zeros] )
        outp_mag = real_outp[:,0]

        ## Get the bin numbers from the true met magnitude (final bin includes overflow)
        bins = T.clamp( targ_mag / (bin_max/n_bins), 0, n_bins-1 ).int().cpu().numpy()

        ## Calculate the batch totals of each metric
        batch_totals = T.zeros( ( len(inputs), 1+len(met_names) ) )
        dot = T.sum( real_outp * real_targ, dim=1 ) / ( outp_mag * targ_mag + 1e-4 )

        batch_totals[:, 0] = T.ones_like(targ_mag)                                             ## Ones are for counting bins
        batch_totals[:, 1] = F.smooth_l1_loss(outputs, targets, reduction="none").mean(dim=1)  ## Loss
        batch_totals[:, 2] = ( ( real_outp - real_targ )**2 ).mean(dim=1)                      ## XY Resolution
        batch_totals[:, 3] = ( outp_mag - targ_mag )**2                                        ## Magnitude Resolution
        batch_totals[:, 4] = T.acos( dot )**2                                                  ## Angular resolution
        batch_totals[:, 5] = ( outp_mag - targ_mag ) / ( targ_mag + 1e-4 )                     ## Deviation from Linearity

        ## Fill in running data by summing over each bin, bin 0 is reserved for dataset totals
        for b in range(n_bins):
            run_totals[b+1] += batch_totals[ bins==b ].sum(axis=0).cpu().numpy()

    ## Include the totals over the whole dataset by summing and placing it in the first location
    run_totals[0] = run_totals.sum(axis=0, keepdims=True)
    run_totals[:,0] = np.clip( run_totals[:,0], 1, None ) ## Just incase some of the bins were empty, dont wana divide by 0

    ## Turn the totals into means or RMSE values
    run_totals[:,1] = run_totals[:,1] / run_totals[:,0]            ## Want average per bin
    run_totals[:,2] = np.sqrt( run_totals[:,2] / run_totals[:,0] ) ## Want RMSE per bin
    run_totals[:,3] = np.sqrt( run_totals[:,3] / run_totals[:,0] ) ## Want RMSE per bin
    run_totals[:,4] = np.sqrt( run_totals[:,4] / run_totals[:,0] ) ## Want RMSE per bin
    run_totals[:,5] = run_totals[:,5] / run_totals[:,0]            ## Want average per bin

    ## Flatten the metrics and drop the number of events in each bin
    metrics = run_totals[:,1:].flatten(order='F')
    metrics = np.expand_dims(metrics, 0)

    ## Getting the names of the columns
    cols = [ met+str(i) for met in met_names for i in range(-1, n_bins) ]

    ## Write the dataframe to the csv
    df = pd.DataFrame( data=metrics, index=["Tight"], columns=cols )
    fnm = ( "Tight_perf.csv" )
    df.to_csv( fnm, mode="w" )

if __name__ == "__main__":
    main()
