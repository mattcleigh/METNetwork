import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt


def main():

    ## The program settings
    dir = "/home/users/l/leighm/METNetwork/METNetwork/output/input_histograms/"
    processes = [
        "ttbar_410470",
        # "llvv_364254",
        "ttbar_345935",
        "ttbar_407345",
        "ttbar_407346",
        "ttbar_407347",
        "all_had_Train_ttbar_410471",
    ]

    ## Create the figure
    fig, ax = plt.subplots()
    ax.set_xlabel("MET GeV")
    ax.set_ylabel("Events")

    ## Cycle through each input process
    base = None
    for proc in processes:

        ## Load the histogram dataframe
        df = pd.read_csv(Path(dir, proc + ".csv"))

        if base is None:
            base = np.zeros_like(df.bins)

        if proc == "all_had_Train_ttbar_410471":
            df.True_ET *= 3000

        ## Plot the histogram
        ax.fill_between(df.bins, base, base + df.True_ET, label=proc)

        base += df.True_ET

    ## Final adjustments and save
    # ax.set_xlim(df.bins[0], df.bins[-1])
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.savefig(Path(dir, "train_combined").with_suffix(".png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
