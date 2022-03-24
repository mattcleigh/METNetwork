from pathlib import Path
import numpy as np
import pandas as pd

from scipy.interpolate import make_interp_spline
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt

dflt_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def contour_with_project(
    input_arrays: np.ndarray, labels: list, ax_labels: list, bins: np.ndarray
):
    """
    Plot a two dimension projection with marginals showing each dimension.
    Can overlay multiple different datasets by passing more than one inputarray argument.
    Each input array needs to have the same corresponding shape for the dimension indices.

    args:
        inputarrays: the two dimensional data to plot
        bins = A 2D array specifying the bin locations
    """

    ## Convert the bins to limits
    xlims = (bins[0][0], bins[0][-1])
    ylims = (bins[1][0], bins[1][-1])
    mid_bins = [(b[1:] + b[:-1]) / 2 for b in bins]

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(7, 2),
        height_ratios=(2, 7),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )

    ## Creating the axis objects
    comb_ax = fig.add_subplot(gs[1, 0])
    marx_ax = fig.add_subplot(gs[0, 0])
    mary_ax = fig.add_subplot(gs[1, 1])

    ## Editing the tick labels and limites
    marx_ax.set_xticklabels([])
    marx_ax.set_yticklabels([])
    mary_ax.set_xticklabels([])
    mary_ax.set_yticklabels([])
    comb_ax.set_xlim(xlims)
    comb_ax.set_ylim(ylims)
    marx_ax.set_xlim(xlims)
    mary_ax.set_ylim(ylims)

    ## Set the axis labels
    comb_ax.set_xlabel(ax_labels[0])
    comb_ax.set_ylabel(ax_labels[1])
    marx_ax.xaxis.set_label_position("top")
    mary_ax.yaxis.set_label_position("right")
    marx_ax.set_xlabel(ax_labels[0] + " marginal")
    mary_ax.set_ylabel(ax_labels[1] + " marginal")

    ## Add the data
    for i, data in enumerate(input_arrays):
        comb_ax.contour(
            *mid_bins, gaussian_filter(data.T, 0.4), colors=dflt_cycle[i], levels=15
        )
        marx_ax.step(
            bins[0],
            [0] + np.sum(data, axis=1).tolist(),
            color=dflt_cycle[i],
            label=labels[i],
        )
        mary_ax.step([0] + np.sum(data, axis=0).tolist(), bins[1], color=dflt_cycle[i])
    marx_ax.legend()

    ## making sure the histograms start at zero (after plotting to autoscale)
    marx_ax.set_ylim(bottom=0)
    mary_ax.set_xlim(left=0)

    return fig


def plot_and_save_contours(
    path: Path,
    hist_list: list,
    labels: list,
    ax_labels: list,
    bins: np.ndarray,
    do_csv: bool = False,
) -> None:
    """Given a list of 2D histograms, an image is saved with the contours and marginals
    superimposed

    args:
        path: Path to save output figure
        hist_list: List of numpy histograms
        labels: Labels for each histogram in hist_list
        ax_labels: Labels for the figure axis
        bins: 2D array of x and y bin edges
    kwargs:
        do_csv: If a csv of the data should be saved as well
    """

    ## Save the histograms to text
    if do_csv:
        mid_bins = [(b[1:] + b[:-1]) / 2 for b in bins]
        np.savetxt(path.with_suffix(".csv"), np.vstack(mid_bins + hist_list))

    ## Create and save contour plots of the histograms
    contour = contour_with_project(hist_list, labels, ax_labels, bins)
    contour.savefig(path.with_suffix(".png"))
    plt.close(contour)


def plot_and_save_hists(
    path: Path,
    hist_list: list,
    labels: list,
    ax_labels: list,
    bins: np.ndarray,
    do_csv: bool = False,
) -> None:
    """Given a list of histograms, an image is saved with them superimposed
    superimposed

    args:
        path: Path to save output figure
        hist_list: List of numpy histograms
        labels: Labels for each histogram in hist_list
        ax_labels: Labels for the figure axis
        bins: Numpy array of bin edges
    kwargs:
        do_csv: If a csv of the data should be saved as well
    """

    ## Save the histograms to text
    if do_csv:
        mid_bins = (bins[1:] + bins[:-1]) / 2
        df = pd.DataFrame(
            np.vstack([mid_bins] + hist_list).T, columns=["bins"] + labels
        )
        df.to_csv(path.with_suffix(".csv"), index=False)

    ## Create the plot of the histograms
    fig, ax = plt.subplots()
    for i, h in enumerate(hist_list):
        ax.step(bins, [0] + h.tolist(), label=labels[i])
    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
