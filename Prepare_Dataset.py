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
    file_core = "*Dataset*1.csv"
    do_rot    = True
    ###################

    ## Collect all locatable files within a dask dataframe
    df = dd.read_csv( os.path.join(base_dir,file_core) )
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

        ## Create a new array based on this rotated dataframe
        arr = df.to_dask_array(lengths=True)

    ## Calculate the mean and deviation on the dataset and convert to an array
    mean = df.mean(axis=0).to_dask_array(lengths=True)
    sdev = df.std(axis=0).to_dask_array(lengths=True)

    ## To normalise we use the array version of the datafile
    normed = (arr - mean) / (sdev+1e-6)

    ## Convert the array back into a dataframe so we can save it
    out_df = normed.to_dask_dataframe(columns=col_names).reset_index()
    out_df.to_hdf( os.path.join( base_dir, "sample-*.h5" ), "/data" )

    ## Package and save the stats together
    stats = np.vstack(( mean.compute(), sdev.compute() ))
    out_file = "stats" + ("_rot" if do_rot else "") + ".csv"
    np.savetxt( os.path.join( base_dir, out_file), stats )

if __name__ == '__main__':
    main()
