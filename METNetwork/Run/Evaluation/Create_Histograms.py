import glob
import numpy as np
import pandas as pd

from pathlib import Path

import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


ProgressBar().register()


class binned_var:
    def __init__(self, name, nbins, range):
        self.name = name
        self.nbins = nbins
        self.range = range
        self.bins = np.linspace(*range, num=nbins + 1)
        self.centers = (self.bins[1:] + self.bins[:-1]) / 2
        self.widths = self.bins[1:] - self.bins[:-1]


def load_dataframe(data_folder, network_names, req_cols, to_GeV=True, run_test=True):
    print("Creating dask dataframe...")

    ## Check if we should reduce to 1 in 10 files
    reduce =  any(a in data_folder for a in ["Zmumu", "ttbar"])
    red_fn = lambda s: "0.t" in s

    ## Load the input files and sort
    inpt_files = glob.glob(data_folder + "*.train-sample.csv")
    if reduce:
        inpt_files = list(filter(red_fn, inpt_files))
    inpt_files.sort()

    ## Load the list of each network file
    netw_files = []
    for nw in network_names:
        nwf = glob.glob(data_folder + "*.train-sample_" + nw + ".csv")
        if reduce:
            nwf = list(filter(red_fn, nwf))
        nwf.sort()
        netw_files.append(nwf)

        if not nwf:
            raise ValueError("No files found for network: {}".format(nw))

    ## Load all the information using dask and convert into a dask array
    inpt_df = dd.read_csv(inpt_files, usecols=req_cols, dtype=np.float32)
    inpt_da = inpt_df.to_dask_array(lengths=True).rechunk(100000, len(req_cols))

    ## Load all of the network estimates into dask as well
    netw_dfs = [dd.read_csv(nwf, dtype=np.float32) for nwf in netw_files]
    netw_das = [nwd.to_dask_array(lengths=True).rechunk(100000, 3) for nwd in netw_dfs]

    ## The arrays are concatenated and converted back into a dask dataframe
    df = dd.from_dask_array(da.concatenate([inpt_da, *netw_das], axis=1))
    df.columns = list(inpt_df.columns) + [c for nwd in netw_dfs for c in nwd.columns]

    ## Change all energy measurements to GeV
    if to_GeV:
        mev_flags = ["_E", "SumET"]
        mev_cols = [col for col in df.columns if any(fl in col for fl in mev_flags)]
        df[mev_cols] /= 1000

    ## Run an assertion test to ensure dataframes are lined up
    if run_test:
        diff = df["True_ET"] - df[f"{network_names[0]}_ETtru"]
        diff = diff.abs().mean().compute()
        print(f"Dataset lineup error is {diff} GeV")

    return df


def add_metric_columns(df, y_list, wp_list):
    """
    Add columns for each y variablie and each working point to the original dataframe
    The y variables are calculated using flags and built in functions that recognise them
    """
    for y in y_list:
        for wp in wp_list:
            name = wp + "_" + y

            if y == "RMSE":
                df[name] = 0.5 * (df[wp + "_EX"] - df["True_EX"]) ** 2
                df[name] += 0.5 * (df[wp + "_EY"] - df["True_EY"]) ** 2

            elif y == "DLin":
                df[name] = df[wp + "_ET"] / (df["True_ET"] + 1e-8) - 1


def add_binned_columns(df, x_list):
    """
    Adds a column to the dataframe showing which bin a certain value falls into
    """
    for x in x_list:
        df[x.name + "_bins"] = (
            df[x.name]
            .map_partitions(pd.cut, x.bins, labels=x.centers)
            .astype(np.float32)
        )


def save_histograms(df, h_list, wp_list, out_hdf):
    """
    Creates histograms for each working point and saves them into a dataframe
    """
    print("Saving histograms...")

    for h in h_list:

        ## For SumET this should only be plotted once!
        if h.name == "Tight_Final_SumET":
            hist, _ = da.histogram(
                df[h.name], bins=h.nbins, range=h.range, density=True
            )
            hist = hist.compute()
            hists = pd.DataFrame(
                data=hist,
                index=h.centers,
                columns=[h.name],
            )

        ## All other histograms are plotted per working point
        else:
            hists = []
            for wp in wp_list:
                hist, _ = da.histogram(
                    df[wp + "_" + h.name], bins=h.nbins, range=h.range, density=True
                )
                hists.append(hist.compute())

            hists = pd.DataFrame(
                data=np.vstack(hists).T,
                index=h.centers,
                columns=[wp + "_" + h.name for wp in wp_list],
            )

        ## Print and save
        key = h.name
        hists.to_hdf(out_hdf, h.name)
        print(" - " + key, "\n")


def save_profiles(df, x_list, y_list, wp_list, out_hdf):
    """
    Create profiles for each working point and saves them into a dataframe
    We use a double loop because it is actually quicker then breaking up the histogram
    """
    print("Saving profiles...")
    for x in x_list:
        for y in y_list:

            ## Only do deviation from linearity with etmiss
            if y == "DLin" and x.name != "True_ET":
                continue

            ycols = [wp + "_" + y for wp in wp_list]
            xcol = [x.name + "_bins"]
            prof = df[ycols + xcol].groupby(xcol).mean().compute()

            ## Apply a square root for all RMSE
            if y == "RMSE":
                prof = da.sqrt(prof)

            key = y + "_vs_" + x.name
            prof.to_hdf(out_hdf, key)
            print(" - " + key, "\n")


def main():

    network_names = ["WithFix"]

    data_base_dir = "/mnt/scratch/Data/METData/Test/"
    hist_out_folder = "/home/users/l/leighm/METNetwork/METNetwork/Output/"

    processes = [
        # ("Z", "user.mleigh.23_02_22.WithSigAndEtaFix.Zmumu_361107_EXT0/"),
        # ("ttbar_a", "user.mleigh.23_02_22.WithSigAndEtaFix.ttbar_410470a_EXT0/"),
        ("ttbar_d", "user.mleigh.23_02_22.WithSigAndEtaFix.ttbar_410470d_EXT0/"),
        # ("WW", "user.mleigh.23_02_22.WithSigAndEtaFix.WW_361600_EXT0/"),
        # ("ZZ", "user.mleigh.23_02_22.WithSigAndEtaFix.ZZ_361604_EXT0/"),
        # ("HW", "user.mleigh.23_02_22.WithSigAndEtaFix.HW_345948_EXT0/"),
        # ("HZ", "user.mleigh.23_02_22.WithSigAndEtaFix.HZ_346600_EXT0/"),
        # ("susy", "user.mleigh.23_02_22.WithSigAndEtaFix.susy_503540_EXT0/"),
    ]

    req_cols = [
        "Tight_Final_ET",
        "Tight_Final_EX",
        "Tight_Final_EY",
        "Tight_Final_SumET",
        "Loose_Final_ET",
        "Loose_Final_EX",
        "Loose_Final_EY",
        "Tghtr_Final_ET",
        "Tghtr_Final_EX",
        "Tghtr_Final_EY",
        "FJVT_Final_ET",
        "FJVT_Final_EX",
        "FJVT_Final_EY",
        "Calo_Final_ET",
        "Calo_Final_EX",
        "Calo_Final_EY",
        "Track_Final_ET",
        "Track_Final_EX",
        "Track_Final_EY",
        "True_ET",
        "True_EX",
        "True_EY",
        "ActMu",
    ]

    ## Register the working points
    wp_list = [
        "Track_Final",
        "Calo_Final",
        "FJVT_Final",
        "Loose_Final",
        "Tight_Final",
        "Tghtr_Final",
        "True",
        *network_names,
    ]

    ## The variables to be binned for histogram comparisons between working points
    h_list = [
        # binned_var("ET", 10, [0, 2000]),
        # binned_var("Tight_Final_SumET", 10, [0, 2500]),
    ]

    ## All of the variables to be binned for the x_axis
    x_list = [
        # binned_var("True_ET", 10, [0, 2000]),
        # binned_var("ActMu", 10, [10, 60]),
        binned_var("Tight_Final_SumET", 10, [200, 1000]),
    ]

    ## All of the variables to plot on the y axis these are just flags
    y_list = [
        "RMSE",
        # "DLin",
    ]

    for proc, folder in processes:

        ## Get the full names of the input and output files
        data_folder = data_base_dir + folder
        out_folder = hist_out_folder + proc
        out_hdf = out_folder + "/hists.h5"

        ## Make sure the output folder exists
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        print(data_folder, out_folder)

        ## Load in all of the information
        df = load_dataframe(data_folder, network_names, req_cols, run_test=False)

        ## Save all the 1 dimensional histograms
        save_histograms(df, h_list, wp_list, out_hdf)

        ## Add columns showing binned information for the x values
        add_binned_columns(df, x_list)

        ## Add columns showing the metrics for the y values
        add_metric_columns(df, y_list, wp_list)

        ## Save all the 2 dimentional profiles
        save_profiles(df, x_list, y_list, wp_list, out_hdf)


if __name__ == "__main__":
    main()
