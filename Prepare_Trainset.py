import os
import h5py
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import dask
import dask.array as da
import dask.dataframe as dd

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
                         required = True )

    return parser.parse_args()

def main():

    args = get_args()

    ## Create the output folder if it doesnt already exist and the file name prefix
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    prefix = "rot_" if args.do_rot else ""

    ## Place all csv files in the folder into a dask dataframe (set blocksize to 16Mb)
    df = dd.read_csv( args.input_dir + "*.csv", assume_missing=True, blocksize=16e6 ) ## Missing means that 0 is a float
    col_names = list(df.columns)

    ## Convert to array the dataframe
    arr = df.to_dask_array()

    if args.do_rot:

        ## Get the names and indices of the elements involved with the rotation
        xcol_names = [ col for col in col_names if "EX" in col ]
        ycol_names = [ col for col in col_names if "EY" in col ]
        xs = np.array([ df.columns.get_loc(col) for col in xcol_names ])
        ys = np.array([ df.columns.get_loc(col) for col in ycol_names ])
        a_idx = df.columns.get_loc("Tight_Phi")

        ## Calculate new rotated elements
        angles = arr[:, a_idx:a_idx+1]
        rotated_x =   arr[:, xs] * da.cos(angles) + arr[:, ys] * da.sin(angles)
        rotated_y = - arr[:, xs] * da.sin(angles) + arr[:, ys] * da.cos(angles)

        ## Convert each of the rotated arrays into a dataframe
        rot_x_df = rotated_x.to_dask_dataframe(columns=xcol_names)
        rot_y_df = rotated_y.to_dask_dataframe(columns=ycol_names)

        ## Replace the appropriate columns in the original df with the corresponding ones in the rotated dfs
        for col in xcol_names:
            df[col] = rot_x_df[col]
        for col in ycol_names:
            df[col] = rot_y_df[col]

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

    ## Create a new array based on the trimmed (maybe rotated) dataframe
    arr = df.to_dask_array(lengths=True)

    ## Calculate the mean and deviation on the dataset and convert to an array
    mean = df.mean(axis=0).to_dask_array(lengths=True)
    sdev = df.std(axis=0).to_dask_array(lengths=True)

    ## To normalise we use the array version of the datafile
    normed = (arr - mean) / (sdev+1e-6)

    ## Convert the array back into a dataframe so we can save it (cast to float)
    out_df = normed.to_dask_dataframe( columns=col_names ).astype(np.float32)
    out_df.to_hdf( os.path.join( args.output_dir, prefix + "sample-*.h5" ), "/data", mode="w" )

    ## Package and save the stats together
    stats = np.vstack(( mean.compute(), sdev.compute() ))
    stat_df = pd.DataFrame( data=stats, columns=col_names )
    stat_df.to_csv( os.path.join( args.output_dir, prefix + "stats.csv" ), index=False )

if __name__ == "__main__":
    main()
