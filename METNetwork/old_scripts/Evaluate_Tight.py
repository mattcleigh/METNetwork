import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

import torch as T
from torch.utils.data import DataLoader

from METNetwork.Resources import Model
from METNetwork.Resources import Utils as myUT
from METNetwork.Resources import Datasets as myDS
from METNetwork.Resources import Plotting as myPL

T.set_grad_enabled(False)
cmap = plt.get_cmap("turbo")

def main():

    inpt_folder = "/mnt/scratch/Data/METData/"
    v_frac = 0.1

    ## Create a dummy model as it has all of the loader capabilities and ensure this is exactly what our networks see during training!
    model = Model.METNET_Agent('Tight', '/mnt/scratch/Saved_Networks/Presentation/')
    model.setup_network(True, 'XXX', None, 1, 5, False, 0, dev='cpu')
    model.inpt_list = ['Tight_Final_ET']
    model.setup_dataset(inpt_folder, v_frac, 32, 1024, 8096, 8, 'mag', 0, 0, 0, no_trn=True)

    ## The bin setup to use for the profiles
    n_bins = 40
    mag_bins = np.linspace(0, 400, n_bins+1)
    exy_bins = [ np.linspace(-50, 250, n_bins+1), np.linspace(-150, 150, n_bins+1) ]

    ## All the networks outputs and targets for the batch will be combined into one list
    all_outputs = []
    all_targets = []

    ## The information to be saved in our dataframe, the truth et (for binning) and the performance metric per bin
    met_names = [ 'Tru', 'Res', 'Lin', 'Ang' ]

    ## Configure pytorch, the network and the loader appropriately
    T.set_grad_enabled(False)
    model.net.eval()
    model.valid_loader.dataset.weight_off()

    ## Iterate through the validation set
    for batch in tqdm(model.valid_loader, desc='perfm', ncols=80, ascii=True):

        ## Get the network outputs and targets
        tight, targets = myUT.move_dev(batch[:-1], model.device)

        ## Undo the processing on Tight
        tight = (tight * model.net.inp_stats[1, :1] + model.net.inp_stats[0, :1]) / 1000
        outputs = T.cat([tight, T.zeros_like(tight)], dim=1)

        all_outputs.append(deepcopy(outputs))
        all_targets.append(deepcopy(targets))

    ## Combine the lists into single tensors
    all_outputs = T.cat(all_outputs)
    all_targets = T.cat(all_targets)

    ## Undo the normalisation on the outputs and the targets
    net_xy = all_outputs
    tru_xy = (all_targets * model.net.trg_stats[1] + model.net.trg_stats[0]) / 1000
    net_et = T.norm(net_xy, dim=1)
    tru_et = T.norm(tru_xy, dim=1)

    ## Calculate the performance metrics
    res = ((net_xy - tru_xy)**2).mean(dim=1)
    lin = (net_et - tru_et) / (tru_et + 1e-8)
    ang = T.acos( T.sum(net_xy*tru_xy, dim=1) / (net_et*tru_et+1e-8) )**2 ## Calculated using the dot product

    ## We save the overall resolution
    model.avg_res = T.sqrt(res.mean()).item()

    ## Combine the performance metrics into a single pandas dataframe
    combined = T.vstack([tru_et, res, lin, ang]).T
    df = pd.DataFrame(myUT.to_np(combined), columns=met_names)

    ## Make the profiles in bins of True ET using pandas cut and groupby methods
    df['TruM'] = pd.cut(df['Tru'], mag_bins, labels=(mag_bins[1:]+mag_bins[:-1])/2)
    profs = df.drop('Tru', axis=1).groupby('TruM', as_index=False).mean()
    profs['Res'] = np.sqrt(profs['Res']) ## Res and Ang are RMSE measurements
    profs['Ang'] = np.sqrt(profs['Ang'])

    ## Save the performance profiles
    profs.to_csv(Path(model.save_dir, model.name, 'perf.csv'), index=False)

    ## Save the Magnitude histograms
    h_tru_et = np.histogram(myUT.to_np(tru_et), mag_bins, density=True)[0]
    h_net_et = np.histogram(myUT.to_np(net_et), mag_bins, density=True)[0]
    myPL.plot_and_save_hists( Path(model.save_dir, model.name, 'MagDist'),
                              [h_tru_et, h_net_et],
                              ['Truth', 'Outputs'],
                              ['MET Magnitude [Gev]', 'Normalised'],
                              mag_bins,
                              do_csv=True )

    ## Save the ex and ey contour plots
    h_tru_xy = np.histogram2d(*myUT.to_np(tru_xy).T, exy_bins, density=True)[0]
    h_net_xy = np.histogram2d(*myUT.to_np(net_xy).T, exy_bins, density=True)[0]
    myPL.plot_and_save_contours( Path(model.save_dir, model.name, 'ExyDist'),
                                 [h_tru_xy, h_net_xy],
                                 ['Truth', 'Outputs'],
                                 ['METx [GeV]', 'METy [GeV]'],
                                 exy_bins,
                                 do_csv=True )

    ## Get a dataframe from the class dict and write out
    dict_df = pd.DataFrame.from_dict([model.get_dict()]).set_index('name')
    dict_df.to_csv(Path(model.save_dir, model.name, 'dict.csv'))

if __name__ == "__main__":
    main()
