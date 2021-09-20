import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

import torch as T
from torch.utils.data import DataLoader

from METNetwork.Resources import Model
from METNetwork.Resources import Datasets as myDS
from METNetwork.Resources import Plotting as myPL

T.set_grad_enabled(False)
cmap = plt.get_cmap("turbo")

def main():

    inpt_folder =  '/mnt/scratch/Data/METData/'
    do_rot = False
    v_frac = 0.1

    ## The bin structure of the histograms
    n_bins = 50
    mag_bins = np.linspace(0, 450, n_bins+1)
    trg_bins = [ np.linspace(-4, 4, n_bins+1)+do_rot, np.linspace(-4, 4, n_bins+1) ]
    exy_bins = [ np.linspace(-400, 400, n_bins+1)+100*do_rot, np.linspace(-400, 400, n_bins+1) ]

    ## Choosing the parameters weighting tests
    weight_tp    = 'trg'
    weight_to    = [ 3.6 ]
    weight_shift = [ 0 ]
    weight_ratio = [ 1 ]

    ## Make all possible test options in a grid
    tests = np.array(np.meshgrid(weight_to, weight_shift, weight_ratio)).T.reshape(-1,3)
    tests = np.concatenate([[[0, 0, 0]], tests]) ## Concatenate truth, untouched!

    ## The list of all the histograms to be filled
    mag_list = []
    trg_list = []
    exy_list = []

    ## Cycle through all possible tests
    for i, (wt, ws, wr) in enumerate(tests):

        ## Create a dummy model as it has all of the loader capabilities and ensure this is exactly what our networks see during training!
        model = Model.METNET_Agent('dummy', '')
        model.setup_network(do_rot, '', None, 1, 5, False, 0, dev='cpu')
        model.inpt_list = ['Tight_Final_ET'] ## Restirct how much data nees to be loaded
        model.setup_dataset(inpt_folder, v_frac, 32, 1024, 8096, 6, weight_tp, wt, ws, wr, no_trn=True)

        ## Initialise the histograms histograms
        mag_hist = np.zeros(n_bins)
        trg_hist = np.zeros((n_bins, n_bins))
        exy_hist = np.zeros((n_bins, n_bins))

        ## Cycle through the dataset
        for (inputs, targets, weights) in tqdm(model.valid_loader):

            ## Un-normalise the targets
            tru_xy = (targets * model.net.trg_stats[1] + model.net.trg_stats[0]) / 1000
            tru_et = T.norm(tru_xy, dim=1)

            ## Convert the required tensors back to numpy arrays
            weights = weights.numpy()
            targets = targets.numpy()
            tru_xy = tru_xy.numpy()
            tru_et = tru_et.numpy()

            ## Update the histograms
            mag_hist += np.histogram(tru_et, mag_bins, weights=weights)[0]
            trg_hist += np.histogram2d(*targets.T, trg_bins, weights=weights)[0]
            exy_hist += np.histogram2d(*tru_xy.T,  exy_bins, weights=weights)[0]

        ## Normalise
        mag_hist /= np.sum(mag_hist)
        trg_hist /= np.sum(trg_hist)
        exy_hist /= np.sum(exy_hist)

        ## Add the histogram to their list
        mag_list.append(mag_hist)
        trg_list.append(trg_hist)
        exy_list.append(exy_hist)

    names = [ 'Truth' ]
    names += [ 'wt={} wr={} ws={}'.format(*cnfg) for cnfg in tests[1:] ]

    ## Save the Magnitude histograms
    myPL.plot_and_save_hists( Path('S_MagDistTR'), mag_list, names, [r'$p_\mathrm{T}^\mathrm{miss}$ [GeV]', 'Normalised'], mag_bins )
    myPL.plot_and_save_contours( Path('S_TrgDistTR'), trg_list, names, ['scaled x', 'scaled y'], trg_bins, do_csv=True )
    myPL.plot_and_save_contours( Path('S_ExyDistTR'), exy_list, names, [r'$p_{x}^\mathrm{miss}$ [GeV]', r'$p_{y}^\mathrm{miss}$ [GeV]'], exy_bins )

if __name__ == "__main__":
    main()
