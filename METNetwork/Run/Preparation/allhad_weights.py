"""
Calculates the weights required in for ttbar allhad to be flat when it is combined
with ttbar non all had
"""


import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

from mattstools.plotting import plot_and_save_hists

def main():

    ## The program settings
    hist_dir = "/home/users/l/leighm/METNetwork/METNetwork/output/input_histograms/"

    ## Load the histograms which define the splines
    non_had = pd.read_csv(Path(hist_dir, hist_dir + "ttbar_410470.csv"))
    all_had = pd.read_csv(Path(hist_dir, hist_dir + "ttbar_410471.csv"))
    comb = non_had.True_ET + all_had.True_ET

    ## Get the original bins not the mid
    bins = non_had.bins

    ## Calculate the bounds of the function
    idx_max = np.argmax(non_had.True_ET)
    y_max = comb[idx_max]

    ## Calculate the sample weighting
    miss = y_max - non_had.True_ET  ## Subtract the non had from the flat to get miss
    weight = miss / all_had.True_ET  ## Use the ratio to give the sampling weight
    weight /= weight[idx_max]  ## Cant have sampling weights above 1
    weight[idx_max:] = 1.0  ## Must not apply sampling after peak

    ## Save the stack plot
    plot_and_save_hists(
        Path(hist_dir, "allhad_with_weights"),
        [non_had.True_ET, weight * all_had.True_ET],
        ["nonhad", "allhad with weights"],
        ["MET [GeV]", "Events"],
        non_had.bins,
        do_csv = True,
        is_mid = True,
        stack = True,
    )

    ## Save the spline function
    plot_and_save_hists(
        Path(hist_dir, "allhad_spline"),
        weight,
        ["allhad sample weights"],
        ["MET [GeV]", "Events"],
        non_had.bins,
        do_csv = True,
        is_mid = True
    )

if __name__ == "__main__":
    main()
