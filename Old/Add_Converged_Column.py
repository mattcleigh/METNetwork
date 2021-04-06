import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def main():

    input_file = "../Grid_Models/*/"

    file_list = glob.glob( input_file )

    ## Create figure
    fig = plt.figure( figsize = (10,5) )
    ax  = fig.add_subplot(111)
    # fig.suptitle()

    best = []
    for f in file_list:
        previous_data = np.loadtxt( Path( f, "train_hist.csv" ) )
        if len(previous_data.shape) < 2:
            continue

        trn_hist = previous_data[:,0].tolist()
        vld_hist = previous_data[:,1].tolist()
        best_epoch = np.argmin(vld_hist) + 1
        epochs_trained = len(trn_hist)
        bad_epochs = len(vld_hist) - best_epoch

        if min(vld_hist) < 0.22 or min(vld_hist) > 0.2340:
            continue

        if Path( f, "perf.csv").is_file():
            ax.plot( vld_hist )
            best.append(f+"perf.csv")

    print(best)
    # plt.legend()
    plt.show()




if __name__ == "__main__":
    main()
