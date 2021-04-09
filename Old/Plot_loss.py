import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def main():

    input_file = "../Grid2_Networks/*/"

    file_list = glob.glob( input_file )

    ## Create figure
    fig = plt.figure( figsize = (10,5) )
    ax  = fig.add_subplot(111)

    for f in file_list:
        previous_data = np.loadtxt( Path( f, "train_hist.csv" ) )
        if len(previous_data.shape) < 2:
            continue

        vld_hist = previous_data[:,1].tolist()

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
