import glob
import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

from Resources import Plotting as myPL
cmap = plt.get_cmap("Spectral")

def main():

    input_search = "Saved_Models/*/perf.csv"

    ## Read csv files
    file_list = glob.glob( input_search )

    df = pd.concat( [ pd.read_csv(f, index_col=0) for f in file_list ] )
    df = df.sort_values( "Res-1", ascending=True )

    ## Calculate the bias and add it as column
    df["bias"] = np.square(df.loc[:, df.columns.str.contains("DLin")].drop(["DLin-1", "DLin0"], axis=1)).mean(axis=1)

    # df = df[ df["depth"] >= 9 ]
    # df = df[ df["width"] == 256 ]
    # df = df[ df["nrm"] == True ]
    # df = df[ df["lr"] == 1e-3 ]
    df = df[ df["do_rot"] == True ]
    df = df[:10]

    # print( df.groupby(['width']).min()["Res-1"])
    # exit()

    x_vals = np.arange(20, 420, 40)                       ## The x-axis bins for the metric plots
    metrics = [ "Loss", "Res", "DLin", "Ang" ]            ## The titles of the metrics to be plotted
    cols    = [ "do_rot", "batch_size", "depth", "width", ## The columns of the coordinate plot
                "skips", "nrm", "lr", "bias", "Loss-1" ]

    ## Make the parallel coord plot
    myPL.parallel_plot( df, cols, "Res-1", curved=True, cmap="Spectral" )

    ## Make the MET plots
    for met in metrics:

        ## Read the appropriate columns
        data = df.loc[ : , df.columns.str.contains(met) ]

        ## Create figure
        fig = plt.figure( figsize = (10,5) )
        ax  = fig.add_subplot(111)
        fig.suptitle(met)

        for i in range(len(df)):
            row = data.iloc[i]
            label = str( list( df.iloc[i][cols[:-2]] ) ) + " " + df.iloc[i].name
            c = cmap( i / len(df) )
            ax.plot( x_vals, row.to_numpy()[1:], "o-", color=c, label=label )
        ax.axhline(0, color="k")
        ax.axvline(0, color="k")
        ax.set_xlim(left=0)
        if met!="DLin":
            ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid()
    plt.show()



if __name__ == "__main__":
    main()
