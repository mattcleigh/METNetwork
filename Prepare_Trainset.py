import time
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.interpolate import interp1d

import dask
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()

def str2bool(v):
    if isinstance(v, bool): return v
    if   v.lower() in ("yes", "true", "t", "y", "1"): return True
    elif v.lower() in ("no", "false", "f", "n", "0"): return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--input_dir",
                         type = str,
                         help = "Folder containing the .csvfiles to be converted",
                         required = True )

    parser.add_argument( "--output_dir",
                         type = str,
                         help = "Destination folder for the .hdf and stat files",
                         required = True )

    parser.add_argument( "--do_rot",
                         type = str2bool,
                         help = "Perform rotations using \"Tight_Phi\"",
                         default = True )

    return parser.parse_args()

def main():

    ## For time testing different methods
    start = time.time()

    ## Get the arguments from the command line
    args = get_args()

    ## Create the path to the output folder and ensure that it is empty (strange things happen with dask to_hdf if they exist)
    output_path = Path( args.output_dir, "Rotated" if args.do_rot else "Raw" )
    if output_path.exists(): shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    ## Read all csv files in the input folder into a dask dataframe (set blocksize to 64Mb)
    df = dd.read_csv( args.input_dir + "*.csv", assume_missing=True, blocksize=64e6 ) ## Missing means that 0 is a float
    col_names = list(df.columns)

    #### We histogram the dataset based on its True ET miss distribution ####
    h_max  = 400e3
    n_bins = 300

    ## Create the histogram
    mags = da.sqrt(df["True_EX"]**2 + df["True_EY"]**2)
    hist, bins = da.histogram(mags, bins=n_bins, range=[0, h_max], density=True )
    mid_bins = ( bins[:-1] + bins[1:] ) / 2
    hist = hist.compute()

    ## Save the histogram as an image and as a csv
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,5) )
    ax1.step( mid_bins, hist, where = "mid" )
    fig.savefig( Path( output_path, "hist.png" ) )
    np.savetxt( Path( output_path, "hist.csv" ), np.vstack((mid_bins, hist)).T, delimiter="," )

    if args.do_rot:

        ## Convert to array the dataframe (chunksize will match blocksize above)
        arr = df.to_dask_array()

        ## Get the names and indices of the elements involved with the rotation
        xcol_names = [ col for col in col_names if "EX" in col ]
        ycol_names = [ col for col in col_names if "EY" in col ]
        xs = [ df.columns.get_loc(col) for col in xcol_names ]
        ys = [ df.columns.get_loc(col) for col in ycol_names ]
        a_idx = df.columns.get_loc("Tight_Phi")

        ## Calculate new rotated elements
        angles = arr[:, a_idx:a_idx+1]
        rotated_x =   arr[:, xs] * da.cos(angles) + arr[:, ys] * da.sin(angles)
        rotated_y = - arr[:, xs] * da.sin(angles) + arr[:, ys] * da.cos(angles)

        ##  Replace the appropriate columns in the original df with the corresponding ones in the rotated dfs
        for i, col in enumerate(xcol_names):
            df[col] = rotated_x[:, i]
        for i, col in enumerate(ycol_names):
            df[col] = rotated_y[:, i]

        ## If we are performing rotations then we dont want Tight EX and EY (drop from df and column names)
        df = df.drop("Tight_Final_EX", axis=1)
        df = df.drop("Tight_Final_EY", axis=1)
        col_names.remove("Tight_Final_EX")
        col_names.remove("Tight_Final_EY")

    ## We drop the angle and DSID, not used in training!
    df = df.drop("Tight_Phi", axis=1)
    df = df.drop("DSID", axis=1)
    col_names.remove("Tight_Phi")
    col_names.remove("DSID")

    ## Calculate the mean and deviation on the dataset and convert to an array
    ## We need to calculate now or face memory issues!
    mean = df.mean(axis=0).compute()
    sdev = df.std(axis=0).compute()

    ## Normalise the dataframe
    normed = (df - mean) / (sdev+1e-6)

    ## After normalisation can add the True ET back in (unnormed), this is needed for weight calculation!
    normed["True_ET"] = mags
    col_names += ["True_ET"]

    ## The column orders change when we normalise (?), we fix this and also cast to float
    normed = normed[col_names].astype("float32")

    ## Save the normalised dataframe to HDF files using the data table
    normed.to_hdf( Path( output_path, "sample-*.h5" ), "data", mode="w", data_columns=None )

    ## Package and save the stats together
    stat_df = pd.concat( [mean, sdev], axis=1 ).transpose()
    stat_df.to_csv( Path( output_path, "stats.csv" ), index=False )

    print(time.time()-start)

if __name__ == "__main__":
    main()
