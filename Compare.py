import glob
import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

from Resources import Plotting as myPL

spec = "turbo"
cmap = plt.get_cmap(spec)

def learning_plot( file_list ):

    ## Create figure
    fig = plt.figure( figsize = (10,5) )
    ax  = fig.add_subplot(111)

    for i, f in enumerate(file_list):
        previous_data = np.loadtxt( f+"train_hist.csv" )
        c = cmap( (i+0.5) / len(file_list) )
        plt.plot( previous_data[:,0].tolist(), "-" , color = c, label=f )
        plt.plot( previous_data[:,1].tolist(), "--", color = c )

    plt.legend()
    plt.show()

def metric_plot( df, metrics, cols ):

    ## Make the MET plots
    for met in metrics:

        ## Read the appropriate columns
        data = df.loc[ : , df.columns.str.contains(met) ]

        ## Create figure
        fig = plt.figure( figsize = (10,5) )
        ax  = fig.add_subplot(111)
        fig.suptitle(met)

        k = 0
        for i in range(len(df))[::-1]:
            x_vals = np.linspace(10, 390, 20) ## Bin centers for the metric plots
            row = data.iloc[i]

            # if df.iloc[i]["weight_to"] == 0:
                # x_vals = np.linspace(20, 380, 10)
                # row = data.iloc[i][:11]

            name = df.iloc[i].name[5:]
            print(name)
            if len(name) != 4 and len(name)!=1:
                continue

            label = name #str( list( df.iloc[i][cols[:-2]] ) )

            ax.plot( x_vals, row.to_numpy()[1:], "x-", color=cmap( (k+0.5) / 4 ), label=label )
            k += 1
            break
        ax.axhline(0, color="k")
        ax.axvline(0, color="k")
        ax.set_xlim(left=0)
        ax.set_ylim(top=0.5)
        ax.set_ylim(bottom=-0.2)
        if met!="DLin":
            ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid()
    plt.show()

def main():

    input_search = "../Saved_Networks/Small_Flat/*/"
    order = "bias"
    N = 0
    restrict = [
                # ( "depth",  9 ),
                # ( "width",  256 ),
                # ( "nrm",    False ),
                # ( "lr",     1e-3 ),
                # ( "do_rot", True )
                ]
    include = [ "Tight_" ]

    ## Find all the networks
    file_list = glob.glob( input_search )

    ## Combine all dataframes, add bias column, invert weight (less twisty), apply restrictions, order df, select best N
    df = pd.concat( [ pd.read_csv(f+"perf.csv", index_col=0) for f in file_list+include ] ).fillna(0)
    df["bias"] = np.square(df.loc[:, df.columns.str.contains("DLin")].drop(["DLin-1", "DLin0"], axis=1)).mean(axis=1)
    df["weight_to"] *= -1
    for flag, value in restrict:
        df = df[ df[flag] == value ]
    df = df.sort_values( order, ascending=True )
    if N != 0: df = df[:N]

    ## Create the parallel coordinate plot
    cols    = [ "do_rot", "batch_size", "depth", "width",
                "skips", "nrm", "lr", "weight_to", "bias", "Loss-1" ]
    # myPL.parallel_plot( df, cols, "Res-1", curved=True, cmap=spec )

    ## Create the metric plots
    metrics = [ "DLin"  ] ## [ "Loss", "Res", "Mag", "Ang", "DLin" ]
    metric_plot( df, metrics, cols )

    ## Make the learning plots
    # learning_plot( file_list )





if __name__ == "__main__":
    main()
