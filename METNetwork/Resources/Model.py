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
import METNetwork.Resources.Networks as myNW
import METNetwork.Resources.Plotting as myPL
import METNetwork.Resources.Datasets as myDS

class METNET_Agent:
    def __init__(self, name, save_dir):
        self.name = name
        self.save_dir = save_dir

    def setup_network(self, do_rot, inpt_rmv, act, depth, width, nrm, drpt, dev='auto'):
        '''
        This initialises the mlp network with the correct size based on the number of parameters
        specified by the input list created in setup_dataset
        '''
        print()
        print('Seting up the neural network')

        ## Update our information dictionary
        self.update_dict(locals())

        ## Creating the neural network with the input list
        self.inpt_list = myUT.setup_input_list(inpt_rmv, do_rot)
        self.net = myNW.METNetwork(self.inpt_list, do_rot, n_out=2, depth=depth, width=width, act_h=act, nrm=nrm, drp=drpt)

        ## Select the device and move the network
        self.device = myUT.sel_device(dev)
        self.net.to(self.device)

    def setup_dataset(self, data_dir, v_frac, n_ofiles, chnk_size, b_size, n_workers, weight_type, weight_to, weight_shift, weight_ratio, no_trn=False):
        '''
        Initialise the train and validation datasets to be used
        '''
        print()
        print('Seting up the datasets')

        ## Update our information dictionary
        self.update_dict(locals())

        ## The base directory of our training data depends on our method of pre-processing
        self.data_dir = Path(data_dir, 'Rotated' if self.do_rot else 'Raw')

        ## Read in the dataset statistics and save them to the network's buffers
        all_stats = T.tensor(pd.read_csv(Path(self.data_dir, 'stats.csv')).to_numpy(), dtype=T.float32, device=self.device)
        self.net.set_statistics(all_stats)

        ## Build the list of files that will be used in the train and validation set
        train_files, valid_files = myDS.buildTrainAndValid(self.data_dir, v_frac)

        ## Get the iterable dataset objects, building the list of columns to read from the HDF files
        dataset_args = (self.inpt_list + ['True_ET', 'True_EX', 'True_EY'], n_ofiles, chnk_size, weight_type, weight_to, weight_shift)
        if no_trn:
            train_files = train_files[:1]
        train_set = myDS.StreamMETDataset(train_files, *dataset_args, weight_ratio)
        valid_set = myDS.StreamMETDataset(valid_files, *dataset_args, 0.0) ## Validation never uses sampling (too noisy)

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

    def setup_training(self, opt_nm, lr, patience, reg_loss_nm, dst_loss_nm, dst_weight, grad_clip):
        '''
        Sets up variables used for training, including the optimiser
        '''
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
        loss_names = ['tot_loss', 'reg_loss', 'dst_loss']
        self.loss_hist = {lnm:{set:[] for set in ['train', 'valid']} for lnm in loss_names }
        self.run_loss = {lnm:myUT.AverageValueMeter() for lnm in loss_names}

        self.num_epochs = 0
        self.best_epoch = 0
        self.bad_epochs = 0

    def run_training_loop(self):
        '''
        This is the main loop which cycles epochs of train and test
        It runs the save method after each epoch and checks for early stopping
        '''
        print()
        print('Running the training loop')

        for epc in count(self.num_epochs+1):
            print( '\nEpoch: {}'.format(epc) )

            ## Run the test/train cycle
            self.epoch(is_train=True)
            self.epoch(is_train=False)

            ## Update the stats
            self.num_epochs += 1
            self.best_epoch = np.argmin(self.loss_hist['tot_loss']['valid']) + 1
            self.bad_epochs = self.num_epochs - self.best_epoch

            ## Save some update
            self.save()

            ## If the total validation loss did not decrease, we check if we have exceeded the patience
            if self.bad_epochs:
                print('Bad Epoch Number: {:}'.format(self.bad_epochs))
                if self.bad_epochs > self.patience:
                    print('Patience Exceeded: Stopping training!')
                    return 0

    def epoch(self, is_train = False):
        '''
        Performs a single epoch on either the train loader or the validation loader.
        Updates the running loss with eeach batch and then the loss history
        '''

        ## Select the correct data, enable/disable sampling, enable/disable gradients, put the network in the correct mode
        if is_train:
            set = 'train'
            loader = self.train_loader
            self.net.train()
            T.set_grad_enabled(True)
        else:
            set = 'valid'
            loader = self.valid_loader
            self.net.eval()
            T.set_grad_enabled(False)

        ## Before each epoch we make sure weighting is enabled and the files are shuffled
        loader.dataset.weight_on()
        loader.dataset.shuffle_files()

        for batch in tqdm(loader, desc=set, ncols=80, ascii=True):

            ## Zero out the gradients
            if is_train:
                self.opt.zero_grad()

            ## Move the batch to the network device and break into parts
            inputs, targets, weights = myUT.move_dev(batch, self.device)

            ## Pass through network
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

            ## Update the each of the running losses
            losses = {'tot_loss':tot_loss, 'reg_loss':reg_loss, 'dst_loss':dst_loss}
            for lnm, obj in self.run_loss.items():
                obj.update(losses[lnm].item())

        ## Use the running losses to update the history and reset
        for lnm, obj in self.run_loss.items():
            self.loss_hist[lnm][set].append(obj.avg)
            obj.reset()

    def save(self):
        '''
        This function saves needed information about the network during training
        Creates a save directory using the model name into which goes
            - /models/   -> A folder containing the network and optimiser versions
            - info.txt   -> A file containing the network setup and description
            - losses.png -> A multiplot showing the loss history per epoch
        When the network validation loss improves the save_perf() method is also called. Check method for details.
        '''

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

        ## Save a file containing the network setup and description (based on get_gict() )
        with open(Path(full_name, 'info.txt'), 'w') as f:
            for k, v in self.get_dict().items():
                f.write('{:15} = {}\n'.format(k, str(v)))
            f.write('\n\n'+str(self.net))
            f.write('\n\n'+str(self.opt))
            f.write('\n') ## One line for whitespace

        ## Save a plot of the loss history
        myPL.plot_multi_loss(Path(full_name, 'losses'), self.loss_hist)

        ## If the network is the best, then we also save the performance file
        if self.bad_epochs==0:
            self.save_perf()

    def load(self, dict_only=False):

        ## Get the name of the directories
        full_name = Path(self.save_dir, self.name)

        ## Load the network
        if not dict_only:
            model_folder = Path(full_name, 'models')
            self.net.load_state_dict(T.load(Path(model_folder, 'net_latest')))
            self.opt.load_state_dict(T.load(Path(model_folder, 'opt_latest')))

        ## Load the previously saved dictionary and use it to update attributes
        old_dict = pd.read_csv(Path(full_name, 'dict.csv')).to_dict('records')[0]
        self.update_dict(old_dict)

    def save_perf(self):
        '''
        This method iterates through the validation set, stores and saves several plots.

        perf.csv: Profiles binned in True ET for the following metrics
            - Resolution (x, y)
            - Deviation from linearity
            - Angular Resolution

        XXXDist.png: Histograms and contours containing
            - 1D distributions of the reconstructed and true et (post-processed)
            - 2D distributions of the reconstructed and true x,y (raw and post-processed)

        dict.csv:
            - All the information from get_dict() stored in one line so we can combine with other networks
        '''
        print()
        print('Running performance profiles')

        ## The bin setup to use for the profiles
        n_bins = 40
        mag_bins = np.linspace(0, 400, n_bins+1)
        trg_bins = [ np.linspace(-3, 5, n_bins+1), np.linspace(-4, 4, n_bins+1) ]
        exy_bins = [ np.linspace(-50, 250, n_bins+1), np.linspace(-150, 150, n_bins+1) ]

        ## All the networks outputs and targets for the batch will be combined into one list
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
        profs['Res'] = np.sqrt(profs['Res']) ## Res and Ang are RMSE measurements
        profs['Ang'] = np.sqrt(profs['Ang'])

        ## Save the performance profiles
        profs.to_csv(Path(self.save_dir, self.name, 'perf.csv'), index=False)

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
        '''
        Creates a dictionary using all strings, intergers and floats from the state dict.
        This should be sufficient for recording all hyperparameters for the network.
        '''
        return { k:str(v) for k, v in self.__dict__.items() if isinstance(v, (str, bool, int, float)) }

    def update_dict(self, vars):
        '''
        Updates the attribute dictionary with new local scope variables.
        Much quicker than simply running self.var = var many times.
        We want local variables to be attributes as the entire dictionary is saved later
        allowing us to quickly store model configuration.
        '''
        self.__dict__.update( (k,v) for k,v in vars.items() if k != 'self' )
