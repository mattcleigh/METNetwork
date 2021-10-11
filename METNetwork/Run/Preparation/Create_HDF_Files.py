import glob
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import dask
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

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

    ## Some dask stuff
    ProgressBar().register()

    ## Get the arguments from the command line
    args = get_args()

    ## Create the path to the output folder and ensure that it is empty (strange things happen with dask to_hdf if they exist)
    output_path = Path(args.output_dir, 'Rotated' if args.do_rot else 'Raw')
    output_path.mkdir(parents=True, exist_ok=True)

    ## Need to delete all HDF files in this folder to prevent partial overwriting
    # for file in output_path.glob('sample-*.h5'):
        # Path(file).unlink()

    ## Work out the number of files being used and therefore the number of partitions needed
    file_search = args.input_dir + '/*/*.train-sample.csv'
    all_files = glob.glob(file_search)
    print('Converting {} csv files'.format(len(all_files)))
    if not all_files:
        raise ValueError('No input files found using: {}'.format(file_search))

    ## Read all csv files in the input folder into a dask dataframe
    print(' -- reading data into a dask dataframe')
    df = dd.read_csv(all_files, dtype=np.float32, blocksize='64MB')
    col_names = list(df.columns)

    ## Remove DSID from the column names as it is not needed here
    df = df.drop('DSID', axis=1)
    col_names.remove('DSID')

    ## Rotate the samples in the dataframe by changing each x and y coordinate
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

        ## Replace the appropriate columns in the original df with the corresponding ones in the rotated dfs
        for i, col in enumerate(xcol_names):
            df[col] = rotated_x[:, i]
        for i, col in enumerate(ycol_names):
            df[col] = rotated_y[:, i]

    ## Get the stats from the entire dataset (compute now to to prevent mem leakage)
    # print(' -- calculating and saving stats')
    # mean = df.mean(axis=0).compute()
    # sdev = df.std(axis=0).compute()
    # stat_df = pd.concat([mean, sdev], axis=1).transpose()
    # stat_df.to_csv(Path(output_path, 'stats.csv'), index=False)

    print(' -- loading previously saved stats')
    stat_df = pd.read_csv(Path(output_path, 'stats.csv'))
    mean = stat_df.iloc[0]
    sdev = stat_df.iloc[1]

    ## Normalise and fix the column orders (they become alphabetical) we also ensure it will be saved as a float
    normed = (df - mean) / (sdev + 1e-6)
    normed = normed[col_names].astype(np.float32)

    ## Add the unnormed True MET back in (used for weighting, only variable not normalised)
    normed['True_ET'] = df['True_ET']

    ## Save the normalised dataframe to HDF files using the data table (columns = True allows us to load specific columns by name)
    # print(' -- creating output files')
    # normed.to_hdf(Path(output_path, 'sample-*.h5'), 'data', mode='w', format='table', data_columns=True)

    ## Create a histogram based on Truth magnitude
    print(' -- creating magnitude histogram')
    mag_bins = np.linspace(0, 450e3, 90+1)
    mag_hist = da.histogram(df['True_ET'], mag_bins, density=True)[0]
    mag_hist = mag_hist.compute()
    myPL.plot_and_save_hists( Path(output_path, 'MagDist'), [mag_hist], ['Truth'],
                              ['MET Magnitude [GeV]', 'Normalised'], mag_bins, do_csv=True )

    ## Create a histogram using the normed targets
    # print(' -- creating target space histogram')
    # trg_bins = [ np.linspace(-4, 4, 50+1)+args.do_rot, np.linspace(-4, 4, 50+1) ]
    # trg_hist = da.histogram2d(normed['True_EX'].to_dask_array(), normed['True_EY'].to_dask_array(), trg_bins, density=True)[0]
    # trg_hist = trg_hist.compute()
    # myPL.plot_and_save_contours( Path(output_path, 'TrgDist'), [trg_hist], ['Truth'],
    #                              ['scaled-x', 'scaled-y'], trg_bins, do_csv=True )


if __name__ == '__main__':
    main()
