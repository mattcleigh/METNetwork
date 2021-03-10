import os
import numpy as np
import pandas as pd

import dask
import dask.array as da
import dask.dataframe as dd

def main():

    ################### User defined variables
    base_dir  = ""
    file_core = "sample*.csv"
    do_rot    = False
    ###################

    ## Collect all locatable files within a dask dataframe
    df = dd.read_csv( os.path.join(base_dir,file_core) )

    ## Calculate the stats of the dataframe
    means  = df.mean(axis=0).compute()
    sdevs  = df.std(axis=0).compute()

    ## If we are doing rotations
    if do_rot:

        ## Convert the dataframe to an array for trig operations
        arr = df.to_dask_array(lengths=True)

        ## Get the indices of the elements involved with the rotation
        xs = np.array([ i for i, col in enumerate(list(df.columns)) if "EX" in col ])
        ys = xs + 1
        a_idx = df.columns.get_loc("Tight_Phi")

        ## Calculate new rotated elements
        angles = arr[:, a_idx:a_idx+1]
        rotated_x =   arr[:, xs] * da.cos(angles) + arr[:, ys] * da.sin(angles)
        rotated_y = - arr[:, xs] * da.sin(angles) + arr[:, ys] * da.cos(angles)

        ## Replace the core means and sdevs with rotated counterparts
        means[xs] = rotated_x.mean(axis=0).compute()
        means[ys] = rotated_y.mean(axis=0).compute()

        sdevs[xs] = rotated_x.std(axis=0).compute()
        sdevs[ys] = rotated_y.std(axis=0).compute()

    print(means)
    print(sdevs)

if __name__ == '__main__':
    main()
