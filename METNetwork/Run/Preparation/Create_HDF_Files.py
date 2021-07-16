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

from METNetwork.Resources import Plotting as myPL

def str2bool(v):
    if isinstance(v, bool): return v
    if   v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument( '--input_dir',
                         type = str,
                         help = 'Folder containing the .csvfiles to be converted',
                         required = True )

    parser.add_argument( '--output_dir',
                         type = str,
                         help = 'Destination folder for the .hdf and stat files',
                         required = True )

    parser.add_argument( '--do_rot',
                         type = str2bool,
                         help = 'Perform rotations using \'Tight_Phi\'',
                         default = True )

    return parser.parse_args()

def main():

    ## Get the arguments from the command line
    args = get_args()

    ## Create the path to the output folder and ensure that it is empty (strange things happen with dask to_hdf if they exist)
    output_path = Path(args.output_dir, 'Rotated' if args.do_rot else 'Raw')
    if output_path.exists(): shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    ## Read all csv files in the input folder into a dask dataframe (set blocksize to 48Mb)
    df = dd.read_csv(args.input_dir + '/*/*.train-sample.csv', assume_missing=True, blocksize=48e6) ## Missing means that 0 is a float
    col_names = list(df.columns)

    ## Create a histogram based on Truth magnitude
    bins = np.linspace(0, 400e3, 80 + 1)
    hist, _ = da.histogram(df['True_ET'], bins, density=True)
    hist = hist.compute()
    myPL.plot_and_save_hists( Path(output_path, 'MagDist'), [hist], ['Truth'],
                              ['MET Magnitude [Gev]', 'Normalised'], bins, do_csv=True )

    if args.do_rot:

        ## Convert to array the dataframe (chunksize will match blocksize above)
        arr = df.to_dask_array()

        ## Get the names and indices of the elements involved with the rotation
        xcol_names = [ col for col in col_names if 'EX' in col ]
        ycol_names = [ col for col in col_names if 'EY' in col ]
        xs = [ df.columns.get_loc(col) for col in xcol_names ]
        ys = [ df.columns.get_loc(col) for col in ycol_names ]
        a_idx = df.columns.get_loc('Tight_Phi')

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
        # df = df.drop('Tight_Final_EX', axis=1)
        # df = df.drop('Tight_Final_EY', axis=1) ## Leave these in for now as they help create the mask for which
        # col_names.remove('Tight_Final_EX')     ## Variables produced by the METNet tool are actually used by the network!!!
        # col_names.remove('Tight_Final_EY')

    ## We drop the angle and DSID, not used in training!
    # df = df.drop('Tight_Phi', axis=1)
    # col_names.remove('Tight_Phi')
    df = df.drop('DSID', axis=1) ## Remove DSID as this is not produced by the METNet tool!
    col_names.remove('DSID')

    ## Calculate the mean and deviation on the dataset and convert to an array
    ## We need to calculate now or face memory issues!
    mean = df.mean(axis=0).compute()
    sdev = df.std(axis=0).compute()

    ## Normalise the dataframe
    normed = (df - mean) / (sdev+1e-6)

    ## Add the unnormed True MET back in (used for weighting)
    normed['True_ET'] = df['True_ET']

    ## The column orders change when we normalise (dont know why), we fix this and cast to float
    normed = normed[col_names].astype('float32')

    ## Save the normalised dataframe to HDF files using the data table
    normed.to_hdf(Path(output_path, 'sample-*.h5'), 'data', mode='w', format='table', data_columns=True)

    ## Package and save the stats together
    stat_df = pd.concat([mean, sdev], axis=1).transpose()
    stat_df.to_csv(Path(output_path, 'stats.csv'), index=False)

    ## Create a histogram using just the normed targets
    trg_bins = [ np.linspace(-3, 5, 40+1), np.linspace(-4, 4, 40+1) ]
    trg_hist = da.histogramdd(normed[['True_EX', 'True_EY']].to_dask_array(), trg_bins, density=True)[0]
    trg_hist = trg_hist.compute()

    ## Save the histogram as an image and as a csv
    myPL.plot_and_save_contours( Path(output_path, 'TrgDist'), [trg_hist], ['Truth'],
                                 ['scaled-x', 'scaled-y'], trg_bins, do_csv=True )

if __name__ == '__main__':
    main()
