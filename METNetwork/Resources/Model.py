import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from itertools import count

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import METNetwork.Resources.Utils as myUT
import METNetwork.Resources.Modules as myML
import METNetwork.Resources.Plotting as myPL
import METNetwork.Resources.Datasets as myDS

class METNET_Agent:
    def __init__(self, name, save_dir):
        self.name = name
        self.save_dir = save_dir

    def setup_network(self, inpt_list, act, depth, width, nrm, drpt, dev='auto'):
        """
        This initialises the mlp network with the correct size based on the number of parameters
        specified by the input list created in setup_dataset
        """
        print()
        print('Seting up the neural network')

        ## Update our information dictionary
        self.update_dict(locals())

        ## Creating the neural network
        self.net = myML.METNetwork( inpt_list, n_out=2, depth=depth, width=width, act_h=act, nrm=nrm, drp=drpt)

        ## Select the device and move the network
        self.device = myUT.sel_device(dev)
        self.net.to(self.device)

    def setup_dataset(self, data_dir, v_frac, n_ofiles, chnk_size, b_size, n_workers, weight_type, weight_to, weight_ratio, weight_shift):
        """
        Initialise the train and validation datasets to be used
        """
        print()
        print('Seting up the datasets')

        ## Update our information dictionary
        self.update_dict(locals())

        ## Read in the dataset statistics and save them to the network's buffers
        all_stats = T.tensor(pd.read_csv(Path(self.data_dir, 'stats.csv')).to_numpy(), dtype=T.float32, device=self.device)
        self.net.set_statistics(all_stats)

        ## Build the list of files that will be used in the train and validation set
        train_files, valid_files = myDS.buildTrainAndValid(data_dir, v_frac)

        ## Get the iterable dataset objects
        dataset_args = (self.inpt_list + ['True_ET', 'True_EX', 'True_EY'], n_ofiles, chnk_size, weight_type, weight_to, weight_ratio, weight_shift)
        train_set = myDS.StreamMETDataset(train_files, *dataset_args)
        valid_set = myDS.StreamMETDataset(valid_files, *dataset_args)

        ## Create the pytorch dataloaders (works for both types of datset)
        loader_kwargs = {'batch_size':b_size, 'num_workers':n_workers, 'drop_last':True, 'pin_memory':True}
        self.train_loader = DataLoader(train_set, **loader_kwargs)
        self.valid_loader = DataLoader(valid_set, **loader_kwargs)

        ## Report on the number of files/samples used
        self.n_train_files = len(train_files)
        self.n_valid_files = len(valid_files)
        self.train_size = len(train_set)
        self.valid_size = len(valid_set)

        print('train set: {:4} files containing {} samples'.format(self.n_train_files, self.train_size))
        print('valid set: {:4} files containing {} samples'.format(self.n_valid_files, self.valid_size))

    def setup_training(self, opt_nm, lr, reg_loss_nm, dst_loss_nm, dst_weight, grad_clip):
        """
        Sets up variables used for training, including the optimiser
        """
        print()
        print('Seting up the training scheme')

        ## Update our information dictionary
        self.update_dict(locals())

        ## Initialise the regression and distribution loss functions
        self.reg_loss_fn = myUT.get_loss(reg_loss_nm)
        self.dst_loss_fn = myUT.get_loss(dst_loss_nm)
        self.do_dst = bool(dst_weight)

        ## Initialise the optimiser
        self.opt = myUT.get_opt(opt_nm, self.net.parameters(), lr)

        ## The history of the losses and their plots
        hist_keys = ['tot_loss', 'reg_loss', 'dst_loss']
        self.history = {prf+key:[] for key in hist_keys for prf in ['train_', 'valid_']}
        self.hist_plots = {key:myPL.loss_plot(ylabel=key) for key in hist_keys}

        self.num_epochs = 0
        self.best_epoch = 0
        self.bad_epochs = 0

    def run_training_loop(self, patience=25):
        """
        This is the main loop which cycles epochs of train and test
        It saves the network and checks for early stopping
        """
        ## Update our information dictionary
        self.update_dict(locals())

        for epc in count(self.num_epochs+1):
            print( '\nEpoch: {}'.format(epc) )

            ## Run the test/train cycle
            self._epoch(is_train=True)
            self._epoch(is_train=False)

            ## At the end of every epoch we save something, even if it is just logging
            self.save()

            ## If the validation loss did not decrease, we check if we have exceeded the patience
            if self.bad_epochs:
                print('Bad Epoch Number: {:}'.format(self.bad_epochs))
                if self.bad_epochs > patience:
                    print('Patience Exceeded: Stopping training!')
                    return 0

        print('\nMax number of epochs completed!')
        return 0

    def _epoch(self, is_train = False):
        """
        This function performs one epoch of training on data provided by the train_loader
        It will also update the graphs after a certain number of batches pass
        """
        ## Put the nework into training mode (for batch_norm/droput)
        if is_train:
            flag = 'train'
            loader = self.train_loader
            self.net.train()
        else:
            flag = 'valid'
            loader = self.valid_loader
            self.net.eval()

        ## Before each epoch we make sure weighting is enabled and the files are shuffled
        T.set_grad_enabled(is_train)
        loader.dataset.weight_on()
        loader.dataset.shuffle_files()

        ## The running losses as a dictionary
        ## These must correspond to the keys in self.history!!!
        ## These must be updated during each batch pass!!!
        run_loss = {'tot_loss':0, 'reg_loss':0, 'dst_loss':0}

        for i, batch in enumerate(tqdm(loader, desc=flag, ncols=80, ascii=True)):

            ## Zero out the gradients
            if is_train:
                self.opt.zero_grad()

            ## Move the batch to the network device and break into parts
            inputs, targets, weights = myUT.move_dev(batch, self.device)

            ## Calculate the network output
            outputs = self.net(inputs)

            ## Calculate the weighted batch regression loss
            reg_loss = (self.reg_loss_fn(outputs, targets).mean(dim=1)*weights).mean()

            ## Calculate the distance matching loss (if required)
            dst_loss = self.dst_loss_fn(outputs, targets) if self.do_dst else T.zeros_like(reg_loss)

            ## Combine the losses
            tot_loss = reg_loss + self.dst_weight * dst_loss

            ## Calculate the gradients and update the parameters
            if is_train:
                tot_loss.backward()
                if self.grad_clip: nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.opt.step()

            ## Update the running losses
            myUT.update_avg_dict(run_loss, (tot_loss.item(), reg_loss.item(), dst_loss.item()), i+1)
            break

        ## Update the history of the network
        self.update_history(flag, run_loss)

    def update_history(self, flag, h_dict):
        """
        Updates the running history network stored in the dictionary self.history
        """
        ## Append the incomming information to the history
        for k, v in h_dict.items():
            self.history[flag+'_'+k].append(v)

        ## Calculate the best epoch and the number of bad epochs (only when 'valid')
        if flag == 'valid':
            self.num_epochs = len(self.history['valid_tot_loss'])
            self.best_epoch = np.argmin(self.history['valid_tot_loss']) + 1
            self.bad_epochs = self.num_epochs - self.best_epoch

    def save(self):
        """
        This function saves needed information about the network during training
        - At the start of training:
            - creates and clears a save directory using the model name
        - Every epoch
            - models   -> A folder containing the network and optimiser versions
            - info.txt -> A file containing the network setup and description
            - history.csv -> Recorded loss history of the training performance
            - history.png -> (Several) plots of the recorded history with their names
        - When the network validation loss improves
            - Runs the save_perf method. Details below
            - perf.csv -> Pandas dataframe performance metrics on the validation set for the best network
            - dict.csv -> Pandas dataframe of the class hyperparameters of the best network (not as readible as info, but can be merged)
            - MagDist.csv -> 1D histogram of the reconstructed and true magnitude (post-processed)
            - TrgDist.csv -> 2D histogram of the reconstructed and true x, y outputs
            - ExyDist.csv -> 2D histogram of the reconstructed and true x, y vectors (post-processed)
        """

        ## The full name of the save directory
        full_name = Path(self.save_dir, self.name)
        full_name.mkdir(parents=True, exist_ok=True)

        ## Save the latest version of the network optimiser (for reloading), and the best network
        model_folder = Path(full_name, 'models')
        model_folder.mkdir(parents=True, exist_ok=True)
        T.save(self.net.state_dict(), Path(model_folder, 'net_latest'))
        T.save(self.opt.state_dict(), Path(model_folder, 'opt_latest'))

        ## For our best network we save the entire model, not just the state dict!!!
        if self.bad_epochs==0:
            T.save(self.net, Path(model_folder, 'net_best'))

        ## Save a file containing the network setup and description (based on class dict)
        with open(Path(full_name, 'info.txt'), 'w') as f:
            for k, v in self.get_dict().items():
                f.write('{:15} = {}\n'.format(k, str(v)))
            f.write('\n\n'+str(self.net))
            f.write('\n\n'+str(self.opt))
            f.write('\n') ## One line for whitespace

        ## Save the loss history using pandas
        hist_frame = pd.DataFrame.from_dict(self.history)
        hist_frame.to_csv(path_or_buf=Path(full_name, 'history.csv'), index=False)

        ## Update and save the loss plots
        for k in self.hist_plots.keys():
            self.hist_plots[k].save( Path(full_name, k+'.png'), self.history['train_'+k], self.history['valid_'+k] )

        ## If the network is the best, then we also save the performance file
        if self.bad_epochs==0:
            self.save_perf()

    def load(self, flag, get_opt=False):

        ## Get the name of the directories
        full_name = Path(self.save_dir, self.name)
        model_folder = Path(full_name, 'models')

        ## Load the network
        if flag == 'best':
            self.net = T.load(Path(model_folder, 'net_best'))
        else:
            self.net.load_state_dict(T.load(Path(model_folder, 'net_'+flag)))

        ## Load the optimiser
        if get_opt:
            self.opt.load_state_dict(T.load(Path(model_folder, 'opt_'+flag)))

        ## Load the previously saved dictionary
        old_dict = pd.read_csv(Path(full_name, 'dict.csv')).to_dict('records')[0]
        self.update_dict(old_dict)

        ## Load the train history
        self.history = pd.read_csv(Path(full_name, 'history.csv')).to_dict(orient='list')
        self.num_epochs = len(self.history['valid_tot_loss'])
        self.best_epoch = np.argmin(self.history['valid_tot_loss']) + 1
        self.bad_epochs = self.num_epochs - self.best_epoch

    def save_perf(self):
        """
        This method iterates through the validation set, and in addition to calculating the loss, it also looks
        at profiles using bins of True ETmiss. This is done both for Tight and for the Network.
        The binned variables are:
            - Resolution (x, y)
            - Deviation from linearity
            - Angular Resolution
        These profiles are combined with information from the class dict to create a performance csv.
        This performance csv only has two lines: the column names and the values. This is so it can be combined
        with results from other networks!
        We also create several a 1D and 2D histograms
            - 1D distributions of the reconstructed and true et (post-processed)
            - 2D distributions of the reconstructed and true x,y (raw and post-processed)
        """
        print('\nImprovement detected! Saving additional information')

        ## The bin setup to use for the profiles
        n_bins = 40
        mag_bins = np.linspace(0, 400, n_bins+1)
        trg_bins = [ np.linspace(-3, 5, n_bins+1), np.linspace(-4, 4, n_bins+1) ]
        exy_bins = [ np.linspace(-50, 250, n_bins+1), np.linspace(-150, 150, n_bins+1) ]

        ## All the networks outputs and targets combined into a single list!
        all_outputs = []
        all_targets = []

        ## The information to be saved in our dataframe, the truth et (for binning) and the performance metric per bin
        met_names = [ 'Tru', 'Res', 'Lin', 'Ang' ]

        ## Configure pytorch, the network and the loader appropriately
        T.set_grad_enabled(False)
        self.net.eval()
        self.valid_loader.dataset.weight_off()

        ## Iterate through the validation set
        for batch in tqdm(self.valid_loader, desc='perfm', ncols=80, ascii=True):

            ## Get the network outputs and targets
            inputs, targets = myUT.move_dev(batch[:-1], self.device)
            outputs = self.net(inputs)

            all_outputs.append(outputs)
            all_targets.append(targets)
            break

        ## Combine the lists into single tensors
        all_outputs = T.cat(all_outputs)
        all_targets = T.cat(all_targets)

        ## Undo the normalisation on the outputs and the targets
        net_xy = (all_outputs * self.net.trg_stats[1] + self.net.trg_stats[0]) / 1000
        tru_xy = (all_targets * self.net.trg_stats[1] + self.net.trg_stats[0]) / 1000
        net_et = T.norm(net_xy, dim=1)
        tru_et = T.norm(tru_xy, dim=1)

        ## Calculate the performance metrics
        res = ((net_xy - tru_xy)**2).mean(dim=1)
        lin = (net_et - tru_et) / (tru_et + 1e-8)
        ang = T.acos( T.sum(net_xy*tru_xy, dim=1) / (net_et*tru_et+1e-8) )**2 ## Calculated using the dot product

        ## We save the overall resolution
        self.avg_res = T.sqrt(res.mean()).item()

        ## Combine the performance metrics into a single pandas dataframe
        combined = T.vstack([tru_et, res, lin, ang]).T
        df = pd.DataFrame(myUT.to_np(combined), columns=met_names)

        ## Make the profiles in bins of True ET using pandas cut and groupby methods
        df['TruM'] = pd.cut(df['Tru'], mag_bins, labels=(mag_bins[1:]+mag_bins[:-1])/2)
        profs = df.drop('Tru', axis=1).groupby('TruM', as_index=False).mean()
        profs['Res'] = np.sqrt(profs['Res'])
        profs['Ang'] = np.sqrt(profs['Ang'])

        ## Save the performance profiles
        profs.to_csv(Path(self.save_dir, self.name, 'perf.csv'), index=False)
        Path(self.save_dir, self.name, 'MagDist.png')

        ## Save the Magnitude histograms
        h_tru_et = np.histogram(myUT.to_np(tru_et), mag_bins, density=True)[0]
        h_net_et = np.histogram(myUT.to_np(net_et), mag_bins, density=True)[0]
        myPL.plot_and_save_hists( Path(self.save_dir, self.name, 'MagDist'),
                                  [h_tru_et, h_net_et],
                                  ['Truth', 'Outputs'],
                                  ['MET Magnitude [Gev]', 'Normalised'],
                                  mag_bins,
                                  do_csv=True )

        ## Save the target contour plots
        h_tru_tg = np.histogram2d(*myUT.to_np(all_targets).T, trg_bins, density=True)[0]
        h_net_tg = np.histogram2d(*myUT.to_np(all_outputs).T, trg_bins, density=True)[0]
        myPL.plot_and_save_contours( Path(self.save_dir, self.name, 'TrgDist'),
                                     [h_tru_tg, h_net_tg],
                                     ['Truth', 'Outputs'],
                                     ['scaled x', 'scaled y'],
                                     trg_bins,
                                     do_csv=True )

        ## Save the ex and ey contour plots
        h_tru_xy = np.histogram2d(*myUT.to_np(tru_xy).T, exy_bins, density=True)[0]
        h_net_xy = np.histogram2d(*myUT.to_np(net_xy).T, exy_bins, density=True)[0]
        myPL.plot_and_save_contours( Path(self.save_dir, self.name, 'ExyDist'),
                                     [h_tru_xy, h_net_xy],
                                     ['Truth', 'Outputs'],
                                     ['METx [GeV]', 'METy [GeV]'],
                                     exy_bins,
                                     do_csv=True )

        ## Get a dataframe from the class dict and write out
        dict_df = pd.DataFrame.from_dict([self.get_dict()]).set_index('name')
        dict_df.to_csv(Path(self.save_dir, self.name, 'dict.csv'))

    def get_dict(self):
        """
        Creates a dictionary using all strings, intergers and floats from the state dict.
        This should be sufficient for recording all hyperparameters for the network.
        """
        return { k:str(v) for k, v in self.__dict__.items() if isinstance(v, (str, bool, int, float)) }

    def update_dict(self, vars):
        """
        Updates the attribute dictionary with new local scope variables (only numbers or strings).
        Much quicker than simply running self.var = var many times.
        We want local variables to be attributes as the entire dictionary is saved later
        allowing us to quickly store model configuration.
        """
        self.__dict__.update( (k,v) for k,v in vars.items() if k != 'self' )
