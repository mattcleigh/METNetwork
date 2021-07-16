import numpy as np
import pandas as pd

from scipy.interpolate import make_interp_spline
from scipy.ndimage.filters import gaussian_filter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

dflt_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

class io_plot:
    """
    A plot that can be updated interactively using the draw and save method
    All inheriting objects need to impliment a _update method
    """
    def draw(self, *args, **kwargs):
        self._update(*args, **kwargs)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, fname, *args, **kwargs):
        self._update(*args, **kwargs)
        self.fig.savefig(fname)

class loss_plot(io_plot):
    """
    A dual plot for train and validation
    """
    def __init__(self, ylabel='', xlabel='epoch'):
        self.fig = plt.figure(figsize=(5,5))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.trn_line, = self.ax.plot([], '-r', label='Train')
        self.vld_line, = self.ax.plot([], '-g', label='Valid')

    def _update(self, trn_data, vld_data, xdata=None):
        self.trn_line.set_data(1+np.arange(len(trn_data)), trn_data)
        self.vld_line.set_data(1+np.arange(len(vld_data)), vld_data)
        self.ax.legend()
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.tight_layout()

def save_hist2D( hist, name, box, ln_coords = [] ):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow( np.log(hist+1), origin='lower', extent = box )
    if len(ln_coords) > 0:
        ax.plot( *ln_coords, 'k-' )
    ax.set_xlabel( 'True MET [GeV]' )
    ax.set_ylabel( 'Network MET [GeV]' )
    plt.tight_layout()
    fig.savefig( name )

def contour_with_project(inputarrays, labels, ax_labels, bins):
    """
    Plot a two dimension projection with marginals showing each dimension.
    Can overlay multiple different datasets by passing more than one inputarray argument.
    Each input array needs to have the same corresponding shape for the dimension indices.

    Inputs:
        inputarray (inputarray2, inputarray3 ...): the two dimensional data to plot
        bins = A 2D array specifying the bin locations
    """

    ## Convert the bins to limits
    xlims = (bins[0][0], bins[0][-1])
    ylims = (bins[1][0], bins[1][-1])
    mid_bins = [(b[1:]+b[:-1])/2 for b in bins]

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec( 2, 2,
                           width_ratios=(7, 2), height_ratios=(2, 7),
                           left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)

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
    marx_ax.set_xlabel(ax_labels[0] + ' marginal')
    mary_ax.set_ylabel(ax_labels[1] + ' marginal')

    ## Add the data
    for i, data in enumerate(inputarrays):
        comb_ax.contour(*mid_bins, gaussian_filter(data.T, 0.7), colors=dflt_cycle[i], levels=10)
        marx_ax.step(bins[0], [0]+np.sum(data, axis=1).tolist(), color=dflt_cycle[i], label=labels[i] )
        mary_ax.step([0]+np.sum(data, axis=0).tolist(), bins[1], color=dflt_cycle[i] )
    marx_ax.legend()

    return fig

def plot_and_save_contours(path, hist_list, labels, ax_labels, bins, do_csv=False):

    mid_bins = [ ( b[1:] + b[:-1] ) / 2 for b in bins ]

    ## Save the histograms to text
    if do_csv:
        np.savetxt(path.with_suffix('.csv'), np.vstack(mid_bins+hist_list))

    ## Create and save contour plots of the histograms
    contour = contour_with_project( hist_list, labels, ax_labels, bins )
    contour.savefig(path.with_suffix('.png'))
    plt.close(contour)

def plot_and_save_hists(path, hist_list, labels, ax_labels, bins, do_csv=False):

    mid_bins = (bins[1:] + bins[:-1]) / 2

    ## Save the histograms to text
    if do_csv:
        df = pd.DataFrame( np.vstack([mid_bins]+hist_list).T, columns=['bins']+labels )
        df.to_csv(path.with_suffix('.csv'), index=False)

    ## Create the plot of the histograms
    fig, ax = plt.subplots()
    for i, h in enumerate(hist_list):
        ax.step(bins, [0] + h.tolist(), label=labels[i])
    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    ax.set_xlim(bins[0], bins[-1])
    ax.legend()
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)

def parallel_plot(df,cols,rank_attr,cmap='Spectral',spread=None,curved=False,curvedextend=0.1):
    """
    Produce a parallel coordinates plot from pandas dataframe with line colour with respect to a column.
    Required Arguments:
        df: dataframe
        cols: columns to use for axes
        rank_attr: attribute to use for ranking
    Options:
        cmap: Colour palette to use for ranking of lines
        spread: Spread to use to separate lines at categorical values
        curved: Spline interpolation along lines
        curvedextend: Fraction extension in y axis, adjust to contain curvature
    Returns:
        x coordinates for axes, y coordinates of all lines
    """
    colmap = matplotlib.cm.get_cmap(cmap)
    cols = cols + [rank_attr]

    fig, axes = plt.subplots(1, len(cols)-1, sharey=False, figsize=(3*len(cols)+3,5))
    valmat = np.ndarray(shape=(len(cols),len(df)))
    x = np.arange(0,len(cols),1)
    ax_info = {}
    for i,col in enumerate(cols):
        vals = df[col]
        if (vals.dtype == float) & (len(np.unique(vals)) > 10):
            minval = np.min(vals)
            maxval = np.max(vals)
            rangeval = maxval - minval
            vals = np.true_divide(vals - minval, maxval-minval)
            nticks = 5
            tick_labels = [round(minval + i*(rangeval/nticks),4) for i in range(nticks+1)]
            ticks = [0 + i*(1.0/nticks) for i in range(nticks+1)]
            valmat[i] = vals
            ax_info[col] = [tick_labels,ticks]
        else:
            vals = vals.astype('category')
            cats = vals.cat.categories
            c_vals = vals.cat.codes
            minval = 0
            maxval = len(cats)-1
            if maxval == 0:
                c_vals = 0.5
            else:
                c_vals = np.true_divide(c_vals - minval, maxval-minval)
            tick_labels = cats
            ticks = np.unique(c_vals)
            ax_info[col] = [tick_labels,ticks]
            if spread is not None:
                offset = np.arange(-1,1,2./(len(c_vals)))*2e-2
                np.random.shuffle(offset)
                c_vals = c_vals + offset
            valmat[i] = c_vals

    extendfrac = curvedextend if curved else 0.05
    for i,ax in enumerate(axes):
        for idx in range(valmat.shape[-1]):
            if curved:
                x_new = np.linspace(0, len(x), len(x)*20)
                a_BSpline = make_interp_spline(x, valmat[:,idx],k=3,bc_type='clamped')
                y_new = a_BSpline(x_new)
                ax.plot(x_new,y_new,color=colmap(valmat[-1,idx]),alpha=0.3)
            else:
                ax.plot(x,valmat[:,idx],color=colmap(valmat[-1,idx]),alpha=0.3)
        ax.set_ylim(0-extendfrac,1+extendfrac)
        ax.set_xlim(i,i+1)

    for dim, (ax,col) in enumerate(zip(axes,cols)):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        ax.yaxis.set_major_locator(ticker.FixedLocator(ax_info[col][1]))

        ## Formatting the tick labels to make them readable
        tick_labels = []
        for a in ax_info[col][0]:
            if isinstance(a, float):
                tick_labels.append('{:.5}'.format(a))
            else:
                tick_labels.append(a)

        ax.set_yticklabels(tick_labels)
        ax.set_xticklabels([cols[dim]])

    plt.subplots_adjust(wspace=0)
    norm = matplotlib.colors.Normalize(0,1)#*axes[-1].get_ylim())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm,pad=0,ticks=ax_info[rank_attr][1],extend='both',extendrect=True,extendfrac=extendfrac)
    if curved:
        cbar.ax.set_ylim(0-curvedextend,1+curvedextend)

    ## Chenge the plot labels to be the configuration, not value
    labels = [ str(row) for row in df[cols[:-3]].values.tolist() ]
    if len(labels) > 10:
        labels = ax_info[rank_attr][0]
    cbar.ax.set_yticklabels(labels)
    cbar.ax.set_xlabel(rank_attr)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, left=0.05, right=0.85)
    plt.show()

    return x,valmat
