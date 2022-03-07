import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path

import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from mattstools.plotting import plot_and_save_hists
from mattstools.utils import sum_other_axes


def main():

    ## The program settings
    outp_dir = "/home/users/l/leighm/METNetwork/METNetwork/output/input_histograms/"
    inpt_dir = "/mnt/scratch/Data/METData/"
    processes = [
        # ("Train", "llvv_364254"),
        # ("Train", "ttbar_345935"),
        # ("Train", "ttbar_407345"),
        # ("Train", "ttbar_407346"),
        # ("Train", "ttbar_407347"),
        # ("Train", "ttbar_410470"),
        # ("Test", "HW_345948"),
        # ("Test", "ZZ_361604"),
        # ("Test", "ttbar_410470a"),
        # ("Test", "HZ_346600"),
        # ("Test", "Zmumu_361107"),
        # ("Test", "ttbar_410470d2"),
        # ("Test", "WW_361600"),
        # ("Test", "susy_503540"),
        # ("all_had", "ttbar_410471"),
        ("all_had/Test", "ttbar_410471"),
        ("all_had/Train", "ttbar_410471"),

    ]
    var_list = [  ## Will plot these on the same axis for now
        "True_ET",
        "Tight_Final_ET",
    ]
    bins = np.linspace(0, 500, 501)
    max_files = -1

    ## Create a dask progress bar
    ProgressBar().register()

    ## Create the output directory
    output_path = Path(outp_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ## Cycle through each input process
    for folder, proc in processes:
        print(proc)

        ## Find the list of all csv files
        inpt_files = list(Path(inpt_dir, folder).glob(f"*{proc}*/*.train-sample.csv"))
        print(f" - found {len(inpt_files)} csv files")

        ## Reduce the number of files down
        if max_files != -1 and max_files < len(inpt_files):
            print(f" - reducing to {max_files}")
            inpt_files = inpt_files[:max_files]

        ## Create a dask dataframe of the csv files
        dataframe = dd.read_csv(
            inpt_files, dtype=np.float32, blocksize="64MB", usecols=var_list
        )

        ## Create a multi histogram so we will only need to run through data once
        dataframe = dataframe[dataframe["True_ET"] > 0]
        hist_2d = da.histogramdd(
            [dataframe[var].to_dask_array() / 1000 for var in var_list],
            bins=[bins, bins],
            normed=False,
        )[0]

        hist_2d = hist_2d.compute()

        ## Sum up the histogram in each dimension and plot
        hist_1d_list = [sum_other_axes(hist_2d, i) for i in range(len(var_list))]

        ## Plot and save the outputs
        plot_and_save_hists(
            Path(outp_dir, folder.replace("/", "_") + "_" + proc),
            hist_1d_list,
            var_list,
            ["MET [GeV]", "Events"],
            bins=bins,
            do_csv=True,
        )


if __name__ == "__main__":
    main()
