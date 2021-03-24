import glob
import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

from Resources import Plotting as myPL

def main():
    cmap = plt.get_cmap('rainbow')

    ## The titles of the metrics to be plotted
    metrics = [ "Loss", "Res", "DLin", "Ang" ]

    ## The columns of the coordinate plot
    cols = [ "do_rot", "batch_size", "depth", "width", "skips", "nrm", "lr", "bias" ]

    ## The x-axis bins for the metric plots
    x_vals = np.arange(20, 420, 40)

    ## Read csv files
    df = pd.concat( [ pd.read_csv(f, index_col=0) for f in glob.glob( "Saved_Models/*/perf.csv" ) ] )

    ## Calculate the bias and add it as column
    df["bias"] = df.loc[:, df.columns.str.contains("Ang")].drop(["Ang-1", "Ang0"], axis=1).mean(axis=1)

    ## Only chose top performing networks
    df = df.sort_values( "Res-1", ascending=True )[:10]

    ## Make the parallel coord plot
    myPL.parallel_plot( df, cols, "Res-1", curved=True )

    ## Make the MET plots
    for met in metrics:

        ## Read the appropriate columns
        data = df.loc[ : , df.columns.str.contains(met) ]

        ## Create figure
        fig = plt.figure( figsize = (5,5) )
        ax  = fig.add_subplot(111)
        fig.suptitle(met)

        for i in range(len(df)):
            row = data.iloc[i]
            label = str( list( df.iloc[i][cols[:-2]] ) )
            c = cmap( i / (len(df)-1) )
            ax.plot( x_vals, row.to_numpy()[1:], "o-", color=c, label=label )
        ax.axhline(0, color="k")
        ax.axvline(0, color="k")
        ax.set_xlim(left=0)
        if met!="Ang":
            ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid()
    plt.show()



if __name__ == "__main__":
    main()
