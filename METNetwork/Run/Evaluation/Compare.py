import glob
import numpy as np
import pandas as pd
import re
import atlasify as at

from pathlib import Path
from pandas.plotting import parallel_coordinates

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from METNetwork.Resources import Plotting as myPL

font = font_manager.FontProperties(family="monospace")

def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

def get_latest_metrics(folder):
    """Loads the latest metrics given a network name
    """
    list_of_folders = glob.glob(folder + "/perf*")
    return max(list_of_folders, key=extract_number)

def make_name(dict_df):
    dict_df["inpt_rmv"] = "All" if dict_df["inpt_rmv"] == "XXX" else "Ind"
    dict_df["dst_weight"] = "D_On" if float(dict_df["dst_weight"]) > 0 else "D_Off"
    dict_df["weight_type"] = (
        dict_df["weight_type"] if float(dict_df["weight_to"]) > 0 else ""
    )
    dict_df["weight_to"] = "W_On" if float(dict_df["weight_to"]) > 0 else "W_Off"
    dict_df["do_rot"] = "R_On" if dict_df["do_rot"] == "True" else "R_Off"
    return dict_df


def join_pad(str_list, pad=6):
    return "".join([x.ljust(pad) for x in str_list])


def histo_plot(net_list, use_labels):

    ## Create the plot of the histograms
    fig, ax = plt.subplots(figsize=(6, 6))

    ## Get the truth and bins from the first element in the list
    bins = np.loadtxt(
        Path(get_latest_metrics(net_list[0]), "MagDist.csv"), usecols=0, skiprows=1, delimiter=","
    )
    truth = np.loadtxt(
        Path(get_latest_metrics(net_list[0]), "MagDist.csv"), usecols=1, skiprows=1, delimiter=","
    )
    ax.plot(bins, truth.tolist(), "-k", label="Truth", linewidth=3)

    for net in net_list:

        histo = np.loadtxt(
            Path(get_latest_metrics(net), "MagDist.csv"), usecols=2, skiprows=1, delimiter=","
        )
        name = Path(net).name
        print(name)
        if use_labels and name != "Tight":
            dict_df = pd.read_csv(net + "/dict.csv", dtype=str)[use_labels]
            name = join_pad(make_name(dict_df.iloc[0]).to_list())
        print(name)
        print()
        fmt = "-r" if name == "Tight" else "-"

        ax.plot(bins, histo.tolist(), fmt, label=name, linewidth=3)

    ## Adjusting the plot
    at.atlasify("Simulation", "Internal\n" "$\sqrt{s}=13$ TeV\n" "$t\overline{t}$")
    ax.set_xlabel(r"$p_\mathrm{T}^\mathrm{miss}$ [GeV]")
    ax.set_ylabel("Normalised")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.legend(prop=font, bbox_to_anchor=(1.00, 1.00), loc="upper right")
    plt.tight_layout()
    fig.savefig("../../Output/Compare_hist.png")


def metric_plot(net_list, use_labels):

    ## Get the list of saved metrics from the first file
    metrics = np.loadtxt(
        get_latest_metrics(net_list[0]) + "/perf.csv", delimiter=",", dtype=str, max_rows=1
    )[1:]

    ## Cycle through the requested metrics
    for met in metrics:

        ## Create figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        ## Cycle through the networks
        for net in net_list:

            ## Get the name of the network to use for the legend
            name = Path(net).name

            if use_labels and name != "Tight":
                dict_df = pd.read_csv(net + "/dict.csv", dtype=str)[use_labels]
                name = join_pad(make_name(dict_df.iloc[0]).to_list())

            ## Load the and plot the performance column
            df = pd.read_csv(get_latest_metrics(net) + "/perf.csv", usecols=["TruM", met])
            fmt = "-ro" if name == "Tight" else "-o"
            ax.plot(df["TruM"], df[met], fmt, label=name)

        ## Work out the limits and labels for the plots
        ax.axhline(0, color="k")
        ax.axvline(0, color="k")
        ax.set_xlim(left=0)
        ax.set_ylim([15, 40])

        if met == "Lin":
            ax.set_ylim(top=0.5)
            ax.set_ylim(bottom=-0.2)

        at.atlasify("Simulation", "Internal\n" "$\sqrt{s}=13$ TeV\n" "$t\overline{t}$")
        ax.set_ylabel(met)
        ax.set_xlabel(r"True $p_\mathrm{T}^\mathrm{miss}$ [GeV]")
        ax.grid()

        ax.legend(prop=font, bbox_to_anchor=(1.00, 1.00), loc="upper right")
        plt.tight_layout()
        fig.savefig("../../Output/Compare_" + met + ".png")


def cutNetList(folder, restr, order, top_n, net_list=[]):

    if not net_list:
        net_list = glob.glob(folder + "*")

    d_list = []
    for n in net_list:
        try:
            d_list += [pd.read_csv(n + "/dict.csv")]
        except:
            pass

    df = pd.concat(d_list)  ## Combine the performance dataframes from each
    for flag, value in restr:
        df = df[
            (df[flag] == value) | (df["name"] == "Tight")
        ]  ## Only show dataframes matching restrictions

    if order:
        df = df.sort_values(
            order, ascending=False
        )  ## Order the dataframes w.r.t. some variable
    if top_n:
        df = df[:top_n][
            ::-1
        ]  ## Reverse the list so that the best networks are plotted last (at the front)

    return [fol + n for (fol, n) in zip(df.save_dir.tolist(), df.name.tolist())]


def main():

    folder = "/mnt/scratch/Saved_Networks/QTPres/"

    order = "avg_res"
    top_n = 0
    restrict = [
        # ( 'weight_to', 0 ),
        # ( 'weight_shift', 0 ),
        # ( 'weight_ratio', 1.0 ),
        # ( 'b_size', 1024 ),
        # ( 'weight_type',  'mag' ),
        # ( 'inpt_rmv',  'Final,_ET' ),
        # ( 'dst_weight',  0.0 ),
        # ( 'do_rot',  True ),
    ]

    use_labels = [
        "do_rot",
        "inpt_rmv",
        "dst_weight",
        "weight_to",
        "weight_ratio",
        "weight_type",
    ]

    net_list = [
        # folder + '53096564_0_15_12_21',
        # folder + '53014519_0_13_12_21',
        # folder + '53014390_0_13_12_21',
        # folder + 'Presentation/Tight',
    ]

    ## Find all the networks and cut the list down based on restrictions
    net_list = cutNetList(folder, restrict, order, top_n, net_list)

    ## Setting up the plotting styles
    # color = plt.cm.rainbow(np.linspace(0, 1, len(net_list)))
    # mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', color)
    at.monkeypatch_axis_labels()
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set3.colors)

    ## Create the metric plots
    metric_plot(net_list, use_labels)

    ## Create the histogram plots
    histo_plot(net_list, use_labels)


if __name__ == "__main__":
    main()
