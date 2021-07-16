import glob
import numpy as np
import pandas as pd

from pathlib import Path
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

from METNetwork.Resources import Plotting as myPL

spec = 'viridis'
cmap = plt.get_cmap(spec)

def metric_plot(net_list):

    ## Get the list of saved metrics from the first file
    metrics = np.loadtxt(net_list[0]+'/perf.csv', delimiter=',', dtype=str, max_rows=1)[1:]

    ## Cycle through the requested metrics
    for met in metrics:

        ## Create figure
        fig = plt.figure( figsize = (10,5) )
        ax = fig.add_subplot(111)

        ## Cycle through the networks
        for net in net_list:

            name = Path(net).name

            ## Load the and plot the performance column
            df = pd.read_csv(net+'/perf.csv', usecols=['TruM', met])
            ax.plot( df['TruM'], df[met], '-o', label=name )

        ## Work out the limits and labels for the plots
        ax.axhline(0, color='k')
        ax.axvline(0, color='k')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        if met=='Lin':
            ax.set_ylim(top=0.5)
            ax.set_ylim(bottom=-0.2)

        ax.set_ylabel(met)
        ax.set_xlabel('True MET [GeV]')
        ax.legend()
        ax.grid()

    plt.show()

def cutNetList(folder, restr, order, top_n):

    net_list = glob.glob(folder+'*')

    df = pd.concat([ pd.read_csv(f+'/dict.csv') for f in net_list ])         ## Combine the performance dataframes from each
    for flag, value in restr: df = df[ df[flag] == value ]        ## Only show dataframes matching restrictions
    if order: df = df.sort_values( order, ascending=True )  ## Order the dataframes w.r.t. some variable
    if top_n: df = df[:top_n]

    return [ folder + n for n in df.name.tolist() ]

def main():

    folder = '/home/matthew/Documents/PhD/Saved_Networks/tmp/'
    order = 'avg_res'
    top_n = 0
    restrict = [
                # ( 'batch_size',  2048 ),
                # ( 'width',  1024 ),
                # ( 'nrm',    True ),
                # ( 'lr',     1e-4 ),
                # ( 'do_rot', True ),
                # ( 'weight_to', -300 ),
                # ( 'weight_ratio', 0.1 ),
                # ( 'weight_shift', 0 ),
                # ( 'skn_weight', 0.1 ),
                # ( 'cut_track', False ),
                # ( 'cut_calo', False ),
                ]

    ## Find all the networks and cut the list down based on restrictions
    net_list = cutNetList(folder, restrict, order, top_n)

    ## Create the metric plots
    metric_plot(net_list)

    ## Do the histogram plot
    # hist_plot( df )





if __name__ == '__main__':
    main()
