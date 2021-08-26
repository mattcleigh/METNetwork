import h5py
import argparse
import numpy as np
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

    return parser.parse_args()

def main():

    ## Get the arguments from the command line
    args = get_args()
    do_rot = ('Rotated' in args.input_dir)

    ## The bins to use
    trg_bins = [ np.linspace(-4, 4, 50+1)+do_rot, np.linspace(-4, 4, 50+1) ]


    ## The dask method
    # ProgressBar().register()
    # df = dd.read_hdf(args.input_dir+'*.h5', 'data', columns=['True_EX', 'True_EY'])
    # print(' -- creating target space histogram')
    # trg_bins = [ np.linspace(-300, 500, 40+1), np.linspace(-400, 400, 40+1) ]
    # trg_hist = da.histogram2d(df['True_EX'].to_dask_array(), df['True_EY'].to_dask_array(), trg_bins, density=True)[0]
    # trg_hist = trg_hist.compute()




    ## The iterative method
    trg_hist = np.zeros((50, 50), dtype=np.float64)
    for i, file in enumerate(Path(args.input_dir).glob('*.h5')):
        hf = h5py.File(file, 'r')
        data = hf['data/table']['True_EX', 'True_EY']
        data = np.array([ list(event) for event in data ]).T
        hf.close()
        trg_hist += np.histogram2d(*data, trg_bins)[0]
        print(i, end='\r')
    print(i)
    trg_hist /= np.sum(trg_hist)




    ## Plot and save
    myPL.plot_and_save_contours( Path(args.input_dir, 'TrgDist'), [trg_hist], ['Truth'],
                                 ['scaled-x', 'scaled-y'], trg_bins, do_csv=True )


if __name__ == '__main__':
    main()
