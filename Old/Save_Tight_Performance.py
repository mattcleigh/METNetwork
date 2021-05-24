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
from Resources import Plotting as myPL

cmap = plt.get_cmap("turbo")

def main():

    inpt_folder =  "../Data/METData/Rotated/"

    ## Defining the histogram
    n_bins = 50
    bin_max = 400

    ## Loading the dataset
    inpt_files = [ f for f in glob.glob( inpt_folder + "*.h5" ) ]
    hist_file = inpt_folder + "hist.csv"

    dataset = myDS.StreamMETDataset( inpt_files, 32, 1024, hist_file, 0, 0, 0 )
    loader = DataLoader( dataset, drop_last=False, batch_size=4096, num_workers=4, pin_memory=True )

    ## Loading the stats
    stat_file = inpt_folder + "stats.csv"
    stats = np.loadtxt(stat_file, skiprows=1, delimiter=",")
    means =  T.from_numpy(stats[0,:])
    devs  =  T.from_numpy(stats[1,:])

    ## The performance metrics to be calculated are stored in a running total matrix
    met_names  = [ "Loss", "Res", "Mag", "Ang", "DLin" ]
    run_totals = np.zeros( ( 1+n_bins, 1+len(met_names) ) ) ## rows = (total, bin1, bin2...) x cols = (n_events, *met_names)

    ## Include the histograms for tight and truth
    tight_h = np.zeros( n_bins )
    truth_h = np.zeros( n_bins )
    truth2D = np.zeros( (n_bins, n_bins) )

    for (inputs, targets, weights) in tqdm( loader, desc="Performance", ncols=80, unit="" ):

        ## Un-normalise the truth targets
        real_targ = ( targets*devs[-2:] + means[-2:] ) / 1000
        targ_mag = T.norm(real_targ, dim=1)

        ## Get the tight met variables
        zeros = T.zeros( [len(inputs), 1] )
        outputs = T.hstack( [inputs[:,:1], zeros] )
        real_outp = ( inputs[:,:1]*devs[0] + means[0] ) / 1000
        real_outp = T.hstack( [real_outp, zeros] )
        outp_mag = real_outp[:,0]

        ## Get the bin numbers from the true met magnitude (final bin includes overflow)
        bins = T.clamp( targ_mag*n_bins/bin_max, 0, n_bins-1 ).int().numpy()

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
            run_totals[b+1] += batch_totals[ bins==b ].sum(axis=0).numpy()

        ## Fill in the reconstructed magnitude histogram
        tight_h += np.histogram( outp_mag, bins=n_bins, range=[0,bin_max] )[0]
        truth_h += np.histogram( targ_mag, bins=n_bins, range=[0,bin_max] )[0]
        truth2D += np.histogram2d( real_targ[:, 1].cpu().tolist(), real_targ[:, 0].cpu().tolist(),
                                   bins=n_bins, range=[[-200, 200], [-100,500]] )[0]

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

    ## Getting the names of the columns and convert the metrics to a dataframe
    mcols = [ met+str(i) for met in met_names for i in range(-1, n_bins) ]
    mdf = pd.DataFrame( data=metrics, index=["Tight"], columns=mcols )

    ## Expand, label and convert the histogram to a dataframe
    hcols = [ "hist"+str(i) for i in range(n_bins) ]
    tight_h = np.expand_dims(tight_h, 0)
    truth_h = np.expand_dims(truth_h, 0)
    tight_df = pd.DataFrame( data=tight_h, index=["Tight"], columns=hcols )
    truth_df = pd.DataFrame( data=truth_h, index=["Truth"], columns=hcols )

    ## Write the combined dataframe to the csv
    tight_df = pd.concat( [ mdf, tight_df ], axis = 1 )
    df = pd.concat( [ tight_df, truth_df ] ).fillna(0) ## Combine the performance dataframe with info on the network

    myPL.save_hist2D( truth2D, "truth2D.png", [-100, 500, -200, 200 ], [[-100,500],[0,0]] )

    df.to_csv( "Tight_perf.csv", mode="w" )

if __name__ == "__main__":
    main()
