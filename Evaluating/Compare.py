import sys
home_env = '../'
sys.path.append(home_env)

import glob
import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

from Resources import Plotting as myPL

spec = "viridis"
cmap = plt.get_cmap(spec)

def learning_plot( file_list ):

    ## Create figure
    fig = plt.figure( figsize = (10,5) )
    ax  = fig.add_subplot(111)

    ## Iterate through the files in the list
    for i, f in enumerate(file_list):
        previous_data = np.loadtxt( f+"train_hist.csv" )
        c = cmap( (i+0.5) / len(file_list) )

        ## Plot the training and validation losses
        plt.plot( previous_data[:,0].tolist(), "-" , color = c )
        plt.plot( previous_data[:,1].tolist(), "--", color = c )

    # plt.legend()
    plt.show()

def hist_plot( df ):

    ## Calculate the bins to use for the histogram
    bins = np.histogram( [], bins=50, range=[0,400] )[1]

    ## Cycle through the samples in the dataframe
    for i, (index, row) in enumerate(df.iterrows()):
        name = index

        ## Pull out only the histogram array from the dataframe row
        hist = row.values[ df.columns.str.contains("hist") ]

        ## The label for the graph is based on the index or the variables
        # if index not in [ "Tight", "Truth" ]:
            # name = "{}_{}".format( row.weight_to, row.skn_weight )

        ## Normalise the histogram
        hist = hist / np.sum(hist)

        ## Get the colour (truth in black) and plot the step function
        c = cmap( (i+1) / (len(df)-2) )
        if index == "Truth": c = "k"
        if index == "Tight": c = "b"
        plt.plot( bins, np.insert(hist,0,0), "-", color=c, label=name )

    ## Setup the the labels and limits of the plot
    plt.ylabel( "Normalised Histogram" )
    plt.xlabel( "MET Magnitude [GeV]" )
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    # plt.yscale('log')
    plt.tight_layout()
    plt.legend()
    plt.show()

def metric_plot( df, metrics, cols ):

    ## The x-axis bin centers for the plots
    x_vals = np.linspace(2, 398, 50)

    ## Cycle through the requested metrics
    for met in metrics:

        ## Create figure
        fig = plt.figure( figsize = (10,5) )
        ax = fig.add_subplot(111)
        fig.suptitle(met)

        ## Cycle through the dataframe
        for i, (index, row) in enumerate(df.iterrows()):
            name = index
            if index == "Truth":
                continue

            ## Pull out the metric array from the row (first is group average)
            vals = row.values[ df.columns.str.contains(met) ][1:]

            # The label for the graph is based on the index or the variables
            if index not in [ "Tight", "Truth" ]:
                name = "{}_{}_{}".format( row.weight_ratio, row.weight_to, row.skn_weight )

            ## Get the colour and make the plot
            c = cmap( (i+1) / (len(df)-2) )
            if index == "Truth": c = "k"
            if index == "Tight": c = "b"
            ax.plot( x_vals, vals, "-o", color=c, label=name )

        ## Work out the limits and labels for the plots
        ax.axhline(0, color="k")
        ax.axvline(0, color="k")
        ax.set_xlim(left=0)
        if met!="DLin":
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(top=0.5)
            ax.set_ylim(bottom=-0.2)
        ax.set_ylabel(met)
        ax.set_xlabel("True MET [GeV]")
        ax.legend()
        ax.grid()
    plt.show()

def main():

    input_search = "../../Saved_Networks/Main/*/"
    order = "Res-1"
    N = 0
    restrict = [
                # ( "batch_size",  2048 ),
                # ( "width",  1024 ),
                # ( "nrm",    True ),
                # ( "lr",     1e-4 ),
                # ( "do_rot", True ),
                # ( "weight_to", -300 ),
                # ( "weight_ratio", 0 ),
                # ( "weight_shift", 0 ),
                # ( "skn_weight", 0 ),
                ]
    include = True

    ## Find all the networks
    file_list = glob.glob( input_search )

    ## Combine all dataframes, add bias column, invert weight (less twisty)
    df = pd.concat( [ pd.read_csv(f+"perf.csv", index_col=0) for f in file_list ] ).fillna(0)
    df["bias"] = np.square(df.loc[:, df.columns.str.contains("DLin")].drop(("DLin"+str(i) for i in range(-1,6)), axis=1)).mean(axis=1)
    df["weight_to"] *= -1

    for flag, value in restrict: df = df[ df[flag] == value ]     ## Only show dataframes matching restrictions
    if order != "": df = df.sort_values( order, ascending=True )  ## Order the dataframes w.r.t. some variable
    if N != 0: df = df[:N]                                        ## Select only the first N variables

    ## Add in tight
    if include:
        df = pd.concat( [ df, pd.read_csv("../Output/Tight_perf.csv", index_col=0) ] ).fillna(0)

    ## Create the parallel coordinate plot
    cols    = [ "weight_to", "weight_shift", "weight_ratio" ]
    # myPL.parallel_plot( df, cols, "bias", curved=True, cmap=spec )

    ## Create the metric plots
    metrics = [ "DLin", "Res", "Ang" ]
    metric_plot( df, metrics, cols )

    ## Make the learning plots
    # learning_plot( file_list )

    ## Do the histogram plot
    hist_plot( df )





if __name__ == "__main__":
    main()
