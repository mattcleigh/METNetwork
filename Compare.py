import glob
import pandas as pd
import matplotlib.pyplot as plt


def main():

    df = pd.concat( [ pd.read_csv(f, index_col=0) for f in glob.glob( "Saved_Models/*/perf.csv" ) ] )

    metrics = [ "Loss", "Res", "DLin", "Ang" ]

    for met in metrics:

        ## Read the appropriate columns
        data = df.loc[ : , df.columns.str.contains(met) ]

        ## Create figure
        fig = plt.figure( figsize = (5,5) )
        ax  = fig.add_subplot(111)
        fig.suptitle(met)

        for idx, row in data.iterrows():
            ax.plot( row.to_numpy()[1:], label=idx )
        plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
