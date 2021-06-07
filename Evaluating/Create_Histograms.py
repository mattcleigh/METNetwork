import sys
home_env = '../'
sys.path.append(home_env)

import os
import glob
import numpy as np
import pandas as pd

import dask
import dask.array as da
import dask.dataframe as df

class binned_var:
    def __init__(self, name, nbins, range):
        self.name = name
        self.nbins = nbins
        self.range = range
        self.bins = np.linspace( *range, num=nbins+1 )
        self.centers = ( self.bins[1:] + self.bins[:-1] ) / 2
        self.widths = self.bins[1:] - self.bins[:-1]

def load_dataframe(data_folder, network_name, req_cols, to_GeV=True):
    print( 'Loading all data into memory' )

    ## Load the input files
    inpt_files = glob.glob( data_folder + '*sample.csv' )
    netw_files = glob.glob( data_folder + '*sample_' + network_name + '.csv' )
    inpt_files.sort()
    netw_files.sort()

    ## Load all the information into memory
    inpt_df = pd.concat( (pd.read_csv(f, dtype=np.float32) for f in inpt_files), ignore_index=True, names=req_cols )
    netw_df = pd.concat( (pd.read_csv(f, dtype=np.float32) for f in netw_files), ignore_index=True )
    df = pd.concat( (inpt_df, netw_df), axis=1 )

    ## Change all measurements to GeV
    if to_GeV:
        mev_flags = ['_E','SumET']
        mev_cols  = [ col for col in df.columns if any( fl in col for fl in mev_flags ) ]
        for col in mev_cols: ## Making this a for loop as we are running out of memory (not much slower)
            df[col] /= 1000
    print(df)
    exit()
    return df


def add_metric_columns(df, y_list, wp_list):
    """
    Add columns for each y variablie and each working point to the original dataframe
    The y variables are calculated using flags and built in functions that recognise them
    """
    for y in y_list:
        for wp in wp_list:
            name = wp+'_'+y

            if y == 'RMSE':
                df[name]  = 0.5 * ( df[wp+'_EX'] - df['True_EX'] )**2
                df[name] += 0.5 * ( df[wp+'_EY'] - df['True_EY'] )**2

            elif y == 'DLin':
                df[name] = df[wp+'_ET'] / ( df['True_ET'] + 1e-8 ) - 1


def add_binned_columns(df, x_list):
    """
    Adds a column to the dataframe showing which bin a certain value falls into
    """
    for x in x_list:
        df[x.name + '_bins'] = pd.cut( df[x.name], x.bins, labels=x.centers ).astype(np.float32)


def save_histograms(df, h_list, wp_list, out_hdf):
    """
    Creates histograms for each working point and saves them into a dataframe
    """
    print('Saving histograms')
    for h in h_list:
        hists = []
        for wp in wp_list:
            bins = pd.cut( df[wp+'_'+h.name], h.bins, labels=h.centers ).astype(np.float32)
            hists.append( bins.groupby(bins).count() / h.widths )
        hists = pd.concat(hists, axis=1)

        key = h.name
        hists.to_hdf( out_hdf, h.name)
        print( ' - ' + key )

def save_profiles(df, x_list, y_list, wp_list, out_hdf):
    """
    Create profiles for each working point and saves them into a dataframe
    We use a double loop because it is actually quicker then breaking up the histogram
    """
    print('Saving profiles')
    for x in x_list:
        for y in y_list:
            ycols = [ wp+'_'+y for wp in wp_list ]
            xcol  = [ x.name + '_bins' ]
            prof = df[ycols+xcol].groupby(xcol).mean()

            ## Apply a square root for all RMSE
            if y == "RMSE":
                prof = np.sqrt(prof)

            key = y + '_vs_' + x.name
            prof.to_hdf( out_hdf, key)
            print( ' - ' + key )

def main():

    network_name = 'FlatSinkhorn'
    data_folder = '../../Data/METData/Raw/ttbar/'

    req_cols = [ 'Tight_Final_ET', 'Tight_Final_EX', 'Tight_Final_EY', 'Tight_Final_SumET',
                 'Loose_Final_ET', 'Loose_Final_EX', 'Loose_Final_EY',
                 'Tghtr_Final_ET', 'Tghtr_Final_EX', 'Tghtr_Final_EY',
                 'FJVT_Final_ET',  'FJVT_Final_EX',  'FJVT_Final_EY',
                 'Calo_Final_ET',  'Calo_Final_EX',  'Calo_Final_EY',
                 'Track_Final_ET', 'Track_Final_EX', 'Track_Final_EY',
                 'True_ET', 'True_EX', 'True_EY',
                 'ActMu' ]

    ## Register the working points
    wp_list = [
                'Track_Final',
                'Calo_Final',
                'FJVT_Final',
                'Loose_Final',
                'Tight_Final',
                'Tghtr_Final',
                'True',
                network_name,
                ]

    ## The variables to be binned for histogram comparisons between working points
    h_list = [
                binned_var( 'ET',  50, [0, 300] ),
             ]


    ## All of the variables to be binned for the x_axis
    x_list  = [
                binned_var( 'True_ET',           50, [0, 300]    ),
                binned_var( 'ActMu',             25, [10, 60]    ),
                binned_var( 'Tight_Final_SumET', 50, [200, 1200] ),
              ]

    ## All of the variables to plot on the y axis these are just flags
    y_list = [
                'RMSE',
                'DLin',
              ]


    ## Setup (and delete existing) the output file
    out_hdf = data_folder + network_name + '_hists.h5'
    if os.path.isfile(out_hdf):
        os.remove(out_hdf)

    df = load_dataframe(data_folder, network_name, req_cols) ## Load in all of the information
    save_histograms(df, h_list, wp_list, out_hdf )           ## Save all the 1 dimensional histograms
    add_binned_columns(df, x_list)                           ## Add columns showing binned information for the x values
    add_metric_columns(df, y_list, wp_list)                  ## Add columns showing the metrics for the y values
    save_profiles(df, x_list, y_list, wp_list, out_hdf )     ## Save all the 2 dimentional profiles

if __name__ == '__main__':
    main()
