import os
import h5py
import time
import numpy as np
import pandas as pd

import dask
import dask.array as da
import dask.dataframe as dd

def main():

    ################### User defined variables
    base_dir  = "Data/Training/"
    file_core = "**train-sample.csv"
    do_rot    = True
    ###################

    ## Collect all locatable files within a dask dataframe
    df = dd.read_csv( os.path.join(base_dir,file_core), assume_missing=True )
    col_names = list(df.columns)

    ## Convert to array the dataframe
    arr = df.to_dask_array(lengths=True)

    if do_rot:
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

        ## Concatenate the rotated elements into one array and convert into a dataframe
        rot_arr = da.concatenate([rotated_x, rotated_y], axis=1)
        rot_df = rot_arr.to_dask_dataframe(columns=xcol_names+ycol_names).reset_index()

        ## Replace the appropriate columns in the original df with the corresponding ones in the rotated df
        for col in xcol_names+ycol_names:
            df[col] = rot_df[col]

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
    outfile = ("rot_" if do_rot else "") + "sample-*.h5"
    out_df.to_hdf( os.path.join( base_dir, outfile ), "/data", mode="w" )

    ## Package and save the stats together
    stats = np.vstack(( mean.compute(), sdev.compute() ))
    stat_df = pd.DataFrame( data=stats, columns=col_names )
    stat_file = ("rot_" if do_rot else "") + "stats.csv"
    stat_df.to_csv( os.path.join( base_dir, stat_file), index=False )

if __name__ == '__main__':
    main()
