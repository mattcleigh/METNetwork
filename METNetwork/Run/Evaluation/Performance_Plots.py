import glob
import scipy.stats
import numpy as np
import pandas as pd
import atlasify as at
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from tqdm import tqdm
from pathlib import Path

from get_bins import get_bins

def get_flag(proc):
    return {
        "susy": r"susy",
        "ttbar_d": r"$t\bar{t}$",
        "WW": r"$WW \rightarrow l\nu l\nu$",
        "ZZ": r"$ZZ \rightarrow ll \nu\nu$",
        "HW": r"$ (VBF) H \rightarrow WW \rightarrow l\nu l\nu$",
        "HZ": r"$(VBF) H \rightarrow ZZ \rightarrow 4\nu$",
        "Z": r"$Z \rightarrow \mu\mu$",
    }[proc]


def finish_step(x_vals, y_vals, hbw):

    ## Remove hbw and add new x value
    x = (x_vals - hbw).tolist()
    x.append(x[-1] + 2 * hbw)

    ## Duplicate final y value
    y = y_vals.tolist()
    y.append(y[-1])

    return x, y


class WP:
    def __init__(self, name, label, colour, fmt):
        self.name = name
        self.label = label
        self.colour = colour
        self.fmt = fmt


class hist_plt:
    def __init__(self, key, xlabel, ylabel, r=None, l=None, t=None, b=None):
        self.key = key
        self.yname = ylabel
        self.xname = key
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.l = l
        self.r = r
        self.t = t
        self.b = b



class prof_plt(hist_plt):
    def __init__(self, xname, yname, *args, **kwargs):
        key = "{}_vs_{}".format(yname, xname)
        super().__init__(key, *args, **kwargs)
        self.xname = xname
        self.yname = yname


def main():

    ## Some plotting configurations
    mpl.rcParams["lines.linewidth"] = 1.5
    at.monkeypatch_axis_labels()

    ## The input arguments
    proc = "ttbar_d"

    ## Some useful strings for plotting
    set = r"Tight $\Sigma p_\mathrm{T}$ [GeV]"
    etm = r"$p_\mathrm{T}^\mathrm{miss}$ [GeV]"
    res = r"$p_{x}^\mathrm{miss}, p_{y}^\mathrm{miss}$ RMSE [GeV]"
    dln = r"$\Delta_\mathrm{T}^\mathrm{lin}$"
    int = r"Interactions per crossing $\langle\mu\rangle$"

    ## Register the working points
    wp_list = [
        WP("True", "True", "black", "-"),
        WP("Track_Final", "Track", "lawngreen", "P"),
        WP("Calo_Final", "Calo", "darkgreen", "d"),
        WP("FJVT_Final", "FJVT", "pink", "s"),
        WP("Loose_Final", "Loose", "cyan", ">"),
        WP("Tight_Final", "Tight", "blue", "<"),
        WP("Tghtr_Final", "Tighter", "darkblue", "^"),
        WP("WithFix", "Network", "red", "d"),
    ]

    ## The variables to be binned for histogram comparisons between working points
    hist_list = [
        hist_plt("Tight_Final_SumET", set, "Normalised Entries"),
        hist_plt("ET", etm, "Normalised Entries"),
    ]

    ## All of the variables to be binned for the x_axis
    prof_list = [
        prof_plt("True_ET", "RMSE", "True " + etm, res),
        prof_plt("ActMu", "RMSE", int, res),
        prof_plt("Tight_Final_SumET", "RMSE", set, res),
        prof_plt("True_ET", "DLin", "True " + etm, dln),
    ]

    ## Open the input files
    folder = "../../Output/" + proc
    file_name = folder + "/hists.h5"
    store = pd.HDFStore(file_name)

    ## Plot the histograms and profiles
    for h in hist_list + prof_list:
        df = store.select(h.key)

        hbw = (df.index[1] - df.index[0]) / 2

        ## Setup the figure and the labels
        fig = plt.figure(figsize=(5, 5))

        if h in hist_list:
            rax = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
            ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        else:
            ax = fig.add_subplot(111)

        ## For SumET we only plot once
        if h.key == "Tight_Final_SumET":
            ax.step(
                *finish_step(df.index, df[h.xname], hbw),
                color="k",
                label="Tight",
                where="post"
            )

        ## Plot each working point line
        else:
            for wp in wp_list:

                if h in hist_list:
                    ax.step(
                        *finish_step(df.index, df[wp.name + "_" + h.xname], hbw),
                        color=wp.colour,
                        label=wp.label,
                        where="post"
                    )
                    rax.plot(
                        df.index,
                        df[wp.name + "_" + h.xname] / df["True_" + h.xname],
                        "-" + wp.fmt,
                        color=wp.colour,
                    )
                else:
                    if wp.name == "True":
                        continue
                    ax.errorbar(
                        df.index,
                        df[wp.name + "_" + h.yname],
                        xerr=hbw,
                        fmt=wp.fmt,
                        color=wp.colour,
                        label=wp.label,
                    )

        ## Final Plotting adjustments
        if h.yname == "DLin":
            ax.axhline(0, color="k")

        at.atlasify("Simulation", "Internal\n" "$\sqrt{s}=13$ TeV\n" + get_flag(proc))
        legend = ax.legend(loc="upper right", fontsize=12)
        legend.get_frame().set_alpha(1)
        legend.get_frame().set_color("white")

        ## Get the bin limits from the store
        left, right, top, bot, _ = get_bins(proc, h.xname, h.yname)
        ax.set_xlim(left=h.l or left, right=h.r or right)
        ax.set_ylim(bottom=h.b or bot, top=h.t or top)

        if h in hist_list:
            ax.set_ylabel(h.ylabel)
            ax.set_xticklabels([])
            rax.set_xlabel(h.xlabel)
            rax.set_ylabel("Ratio to True")
            rax.set_xlim(left=h.l, right=h.r)
            rax.set_ylim(bottom=0.3, top=1.7)
            rax.grid(axis="y")
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.1)
        else:
            ax.set_xlabel(h.xlabel)
            ax.set_ylabel(h.ylabel)
            fig.tight_layout()
        ax.yaxis.label.set_size(15)
        ax.xaxis.label.set_size(15)
        fig.savefig(Path(folder, h.key + ".png"))


if __name__ == "__main__":
    main()
