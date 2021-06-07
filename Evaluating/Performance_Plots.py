import sys
home_env = '../'
sys.path.append(home_env)

import glob
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns
import atlasify as at
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm

class WP:
    def __init__(self, name, label, colour, fmt):
        self.name   = name
        self.label  = label
        self.colour = colour
        self.fmt    = fmt

class hist_plt:
    def __init__(self, key, xlabel, ylabel, r=None, l=None, t=None, b=None):
        self.key = key
        self.yname = key
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.r = r
        self.l = l
        self.t = t
        self.b = b

class prof_plt(hist_plt):
    def __init__(self, xname, yname, *args, **kwargs):
        key = '{}_vs_{}'.format(yname, xname)
        super().__init__(key, *args, **kwargs)
        self.xname = xname
        self.yname = yname

def main():

    ## Setting up some plotting parameters
    at.monkeypatch_axis_labels()
    # mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    # mpl.rcParams['font.family'] = 'Times New Roman'

    network_name = 'FlatSinkhorn'
    data_folder = '../../Data/METData/Raw/ttbar/'
    process = r'$t\bar{t}$'

    ## Some useful strings for plotting
    set = r'Tight $\Sigma E_\mathrm{T}$ [GeV]'
    etm = r'$E_\mathrm{T}^\mathrm{miss}$ [GeV]'
    res = r'$E_{x}^\mathrm{miss}, $E_{x}^\mathrm{miss}$ RMSE [GeV]'
    dln = r'$\Delta_\mathrm{T}^\mathrm{lin}$'

    ## Register the working points
    wp_list = [
                WP('Track_Final', 'Track',   'brown',    'P'),
                WP('Calo_Final',  'Calo',    'green',    'd'),
                # WP('FJVT_Final',  'FJVT',    'pink',     's'),
                WP('Loose_Final', 'Loose',   'cyan',     '>'),
                WP('Tight_Final', 'Tight',   'blue',     '<'),
                # WP('Tghtr_Final', 'Tigher',  'darkblue', '^'),
                WP(network_name,  'Network', 'red',      'o'),
                WP('True',        'True',    'black',    '-'),
                ]

    ## The variables to be binned for histogram comparisons between working points
    hist_list = [
        # hist_plt( 'ET', etm, 'Entries / 6 GeV', l=0, r=300, b=0, t=38000 )
             ]


    ## All of the variables to be binned for the x_axis
    prof_list = [
        # prof_plt( 'True_ET',           'RMSE', 'True '+etm, res, b=20, t=70),
        # prof_plt( 'ActMu',             'RMSE', 'Interactions per crossing $\langle\mu\rangle$', res, b=20, t=70),
        # prof_plt( 'Tight_Final_SumET', 'RMSE',  set, res, b=20, t=70),
        prof_plt( 'True_ET',           'DLin', 'True '+etm, dln, b=-0.5, t=1, l=0, r=300),
              ]

    file_name = data_folder + network_name + '_hists.h5'
    store = pd.HDFStore(file_name)

    ## Plot the histograms and profiles
    for h in hist_list+prof_list:
        df = store.select(h.key)

        ## Setup the figure and the labels
        fig = plt.figure( figsize = (5,5) )

        if h in hist_list:
            rax = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
            ax  = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        else:
            ax  = fig.add_subplot(111)


        ## Plot each working point line
        for wp in wp_list:
            if wp.name == 'True' and h in prof_list: continue
            ax.plot( df.index, df[wp.name+'_'+h.yname], wp.fmt, color=wp.colour, label=wp.label )

            if h in hist_list:
                rax.plot( df.index, df[wp.name+'_'+h.yname] / df['True_'+h.yname], wp.fmt, color=wp.colour )

        ## Final Plotting adjustments
        if h.yname == 'DLin':
            ax.axhline(0, color='k')

        ax.legend(loc='upper right', fontsize=12)
        at.atlasify('Simulation', 'work in progress\n' '$\sqrt{s}=13$ TeV\n' + process)
        ax.set_xlim(left=h.l, right=h.r)
        ax.set_ylim(bottom=h.b, top=h.t)

        if h in hist_list:
            ax.set_ylabel(h.ylabel)
            ax.set_xticklabels([])
            rax.set_xlabel(h.xlabel)
            rax.set_ylabel('Ratio to True')
            rax.set_xlim(left=h.l, right=h.r)
            rax.grid(axis='y')
            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
        else:
            ax.set_xlabel(h.xlabel)
            ax.set_ylabel(h.ylabel)
            fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
