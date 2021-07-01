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
import METNetwork.Resources.Plotting as myPL
import METNetwork.Resources.Datasets as myDS

class METNET_Agent:
    def __init__(self, name, save_dir):
        self.name = name
        self.save_dir = save_dir

    def setup_dataset(self, inpt_list, data_dir, v_frac, n_ofiles, chnk_size, weight_to, weight_ratio, weight_shift):
        """
        Initialise the train and validation datasets to be used
        """
        print()
        print('Seting up the datasets')

        ## Update our information dictionary
        self.update_dict(locals())

        ## Build the list of files that will be used in the train and validation set
        train_files, valid_files = myDS.buildTrainAndValid(data_dir, v_frac)

        ## Get the iterable dataset objects
        dataset_args = (inpt_list, n_ofiles, chnk_size, weight_to, weight_ratio, weight_shift)
        self.train_set = myDS.StreamMETDataset(train_files, *dataset_args)
        self.valid_set = myDS.StreamMETDataset(valid_files, *dataset_args)

        ## Report on the number of files/samples used
        self.n_train_files = len(train_files)
        self.n_valid_files = len(valid_files)
        self.train_size = len(self.train_set)
        self.valid_size = len(self.valid_set)

        print('train set: {:4} files containing {} samples'.format(self.n_train_files, self.train_size))
        print('valis set: {:4} files containing {} samples'.format(self.n_valid_files, self.valid_size))

    def setup_network(self, act, depth, width, nrm, drpt):
        """
        This initialises the mlp network with the correct size based on the number of parameters
        specified by the input list created in setup_dataset
        """
        print()
        print('Seting up the neural network')

        ## Update our information dictionary
        self.update_dict(locals())

        ## Creating the neural network
        self.net = myUT.mlp_creator( n_in=len(self.inpt_list), n_out=2,
                                     depth=depth, width=width, act_h=act, nrm=nrm, drp=drpt)

        ## Save and register the statistics so they can be loaded with the network
        stats = T.tensor(np.loadtxt(Path(self.data_dir, 'stats.csv'), skiprows=1, delimiter=','))
        self.net.register_buffer('preproc_stats', stats)

        ## Select the device and move the network
        self.device = myUT.sel_device('cpu')
        self.net.to(self.device)
        print(self.net)

    def setup_training(self, loss_nm, opt_nm, lr, grad_clip, skn_weight, b_size, n_workers):
        """
        Sets up variables used for training, including the optimiser
        """
        print()
        print('Seting up the training scheme')

        ## Update our information dictionary
        self.update_dict(locals())

        ## Initialise the regression and distribution loss functions
        self.rec_loss_fn = myUT.get_loss(loss_nm)
        self.skn_loss_fn = myUT.get_loss('snkhrn')
        self.do_skn = bool(skn_weight)

        ## Initialise the optimiser
        self.opt = myUT.get_opt(opt_nm, self.net.parameters(), lr)

        ## Create the pytorch dataloaders (works for both types of datset)
        loader_kwargs = {'batch_size':b_size, 'num_workers':n_workers, 'drop_last':True, 'pin_memory':True}
        self.train_loader = DataLoader(self.train_set, **loader_kwargs)
        self.valid_loader = DataLoader(self.valid_set, **loader_kwargs)

        ## The history of the losses and their plots
        hist_keys = ['tot_loss', 'rec_loss', 'skn_loss']
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

    def _epoch( self, is_train = False ):
        """
        This function performs one epoch of training on data provided by the train_loader
        It will also update the graphs after a certain number of batches pass
        """
        ## Put the nework into training mode (for batch_norm/droput)
        if is_train:
            flag = 'train'
            loader = self.train_loader
            T.enable_grad()
            self.net.train()
        else:
            flag = 'valid'
            loader = self.valid_loader
            T.no_grad()
            self.net.eval()

        ## Before each epoch we must shuffle the files order
        loader.dataset.shuffle_files()

        ## The running losses
        rn_tot = 0
        rn_rec = 0
        rn_skn = 0

        for i, batch in enumerate(tqdm(loader, desc=flag, ncols=100, ascii=True)):
            print(batch)
            exit()
            ## Zero out the gradients
            if is_train:
                self.opt.zero_grad()

            ## Move the batch to the network device and break into parts
            inputs, targets, weights = myUT.move_dev(batch, self.device)

            ## Calculate the network output
            outputs = self.net(inputs)

            ## Calculate the weighted batch reconstruction loss
            rec_loss = (self.loss_fn(outputs, targets).mean(dim=1)*weights).mean()

            ## Calculate the sinkhorn loss (if required)
            skn_loss = self.dist_loss(outputs, targets) if self.do_skn else T.zeros_like(rec_loss)

            ## Combine the losses
            tot_loss = rec_loss + self.skn_weight * skn_loss

            ## Calculate the gradients and update the parameters
            if is_train:
                tot_loss.backward()
                if self.grad_clip: nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.opt.step()

            ## Update the running losses
            rn_tot = myUT.update_avg(rn_tot, tot_loss, i+1)
            rn_rec = myUT.update_avg(rn_rec, rec_loss, i+1)
            rn_skn = myUT.update_avg(rn_skn, skn_loss, i+1)

        ## Update the history of the network
        self.update_history(flag, {'tot_loss':rn_tot, 'rec_loss':rn_rec, 'skn_loss':rn_skn})

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
            - loss.csv -> Recorded loss history of the training performance
            - loss.png -> (Several) plots of the recorded history
        - When the network validation loss improves
            - perf.csv -> Pandas dataframe Performance metrics on the validation set for the best network
            - MagHist.csv -> 1D histogram of the reconstructed, tight and true magnitude
            - ExyHist.csv -> 2D histogram of the reconstructed, tight and true x, y outputs
        """

        ## The full name of the save directory
        full_name = Path(self.save_dir, self.name)
        full_name.mkdir(parents=True, exist_ok=True)

        ## Save the latest version of the network optimiser (for reloading), and the best network
        model_folder = Path(full_name, 'models')
        model_folder.mkdir(parents=True, exist_ok=True)
        T.save(self.net.state_dict(), Path(model_folder, 'net_latest'))
        T.save(self.opt.state_dict(), Path(model_folder, 'opt_latest'))
        if self.bad_epochs==0:
            T.save(self.net.state_dict(), Path(model_folder, 'net_best'))

        ## Save a file containing the network setup and description (based on class dict)
        with open(Path(full_name, 'info.txt'), 'w') as f:
            for k, v in self.__dict__.items():
                if len(str(v)) > 50: continue ## Save shorter stings
                f.write('{:15} = {}\n'.format(k, str(v)))
            f.write('\n\n'+str(self.network))
            f.write('\n\n'+str(self.opt))
            f.write('\n') ## One line for whitespace

        ## Save the loss history using pandas
        hist_frame = pd.DataFrame.from_dict(self.history)
        hist_frame.to_csv(path_or_buf=Path(full_name, 'hist.csv'), index=False)

        ## Update and save the loss plots
        for k in self.hist_plots.keys():
            self.hist_plots[k].save( Path(full_name, k+'.png'), self.history['train_'+k], self.history['valid_'+k] )

        ## If the network is the best, then we also save the performance file
        if self.bad_epochs==0:
            self.save_perf()

    def load(self, flag):

        ## Get the name of the directories
        full_name = Path(self.save_dir, self.name)
        model_folder = Path(full_name, 'models')

        ## Load the network and optimiser
        self.net.load_state_dict(T.load(Path(model_folder, 'net_'+flag)))
        if flag == 'latest':
            self.opt.load_state_dict(T.load(Path(model_folder, 'opt_'+flag)))

        ## Load the train history
        self.history = pd.read_csv(Path(full_name, 'hist.csv')).to_dict(orient='list')
        self.num_epochs = len(self.history['valid_loss'])
        self.best_epoch = np.argmin(self.history['valid_loss']) + 1
        self.bad_epochs = self.num_epochs - self.best_epoch

    def save_perf(self):
        """
        This method iterates through the validation set, and in addition to calculating the loss, it also looks
        at profiles using bins of True ETmiss.
        The binned variables are:
            - Reconstruction loss
            - Sinkhorn loss
            - Resolution (x, y)
            - Andular Resolution
            - Deviation from linearity
        These profiles are combined with infromation from the class dict to create a performance csv
        We also create several a 1D and 2D histograms
            - 1D histogram of the reconstructed, tight and true magnitude
            - 2D histogram of the reconstructed, tight and true x, y outputs
        """
        print('\nImprovement detected! Saving additional information')

        ## The bin setup to use for the profiles
        n_bins = 25
        bin_max = 400 ## In GeV

        ## The performance metrics to be calculated are stored in a running total matrix
        met_names  = [ 'Res', 'Rec', 'Skn', 'Ang', 'Lin' ]
        run_totals = np.zeros((1+n_bins, 1+len(met_names))) ## rows = (total, bin1, bin2...) x cols = (n_events, *met_names)

        ## Build a 1D and 2D histogram of the output magnitude
        hist1D = np.zeros(n_bins)
        hist2D = np.zeros((n_bins, n_bins))
        outp2D = np.zeros((n_bins, n_bins))

        ## Iterate through the validation set
        T.no_grad()
        self.network.eval()
        self.valid_loader.dataset.weight_off()

        for batch in tqdm(self.valid_loader, desc='perfm', ncols=100, ascii=True):

            ## Move the batch to the network device and break into parts
            inputs, targets, weights = myUT.move_dev(batch, self.device)
            outputs = self.net(inputs)

            ## Un-normalise the outputs and the targets
            real_outp = (outputs*devs + means) / 1000 ## Convert to GeV (Dont like big numbers)
            real_targ = (targets*devs + means) / 1000

            ## Calculate the magnitudes of the vectors
            targ_mag = T.norm(real_targ, dim=1)
            outp_mag = T.norm(real_outp, dim=1)

            ## Get the bin numbers from the true met magnitude (final bin includes overflow)
            bins = T.clamp( targ_mag * self.n_bins / self.bin_max, 0, self.n_bins-1 ).int().cpu().numpy()

            ## Calculate the batch totals of each metric
            batch_totals = T.zeros( ( len(inputs), 1+len(met_names) ) )
            dot = T.sum( real_outp * real_targ, dim=1 ) / ( outp_mag * targ_mag + 1e-4 )

            batch_totals[:, 0] = T.ones_like(targ_mag)                                             ## Ones are for counting bins
            batch_totals[:, 1] = F.smooth_l1_loss(outputs, targets, reduction='none').mean(dim=1)  ## Loss
            batch_totals[:, 2] = ( ( real_outp - real_targ )**2 ).mean(dim=1)                      ## XY Resolution
            batch_totals[:, 3] = ( outp_mag - targ_mag )**2                                        ## Magnitude Resolution
            batch_totals[:, 4] = T.acos( dot )**2                                                  ## Angular resolution
            batch_totals[:, 5] = ( outp_mag - targ_mag ) / ( targ_mag + 1e-4 )                     ## Deviation from Linearity

            ## Fill in running data by summing over each bin, bin 0 is reserved for dataset totals
            for b in range(self.n_bins):
                run_totals[b+1] += batch_totals[ bins==b ].sum(axis=0).cpu().numpy()

            ## Fill in the magnitude histograms
            hist1D += np.histogram( outp_mag.cpu().tolist(), bins=self.n_bins, range=[0,self.bin_max] )[0]
            hist2D += np.histogram2d( outp_mag.cpu().tolist(), targ_mag.cpu().tolist(),
                                      bins=self.n_bins, range=[[0,self.bin_max2D], [0,self.bin_max2D]] )[0]
            outp2D += np.histogram2d( real_outp[:, 1].cpu().tolist(), real_outp[:, 0].cpu().tolist(),
                                      bins=self.n_bins, range=[[-200, 200], [-200,400]] )[0]

        ## Include the totals over the whole dataset by summing and placing it in the first location
        run_totals[0] = run_totals.sum(axis=0, keepdims=True)
        run_totals[:,0] = np.clip( run_totals[:,0], 1, None ) ## Just incase some of the bins were empty, dont wana divide by 0

        ## Turn the totals into means or RMSE values
        run_totals[:,1] = run_totals[:,1] / run_totals[:,0]            ## Want average per bin
        run_totals[:,2] = np.sqrt( run_totals[:,2] / run_totals[:,0] ) ## Want RMSE per bin
        run_totals[:,3] = np.sqrt( run_totals[:,3] / run_totals[:,0] ) ## Want RMSE per bin
        run_totals[:,4] = np.sqrt( run_totals[:,4] / run_totals[:,0] ) ## Want RMSE per bin
        run_totals[:,5] = run_totals[:,5] / run_totals[:,0]            ## Want average per bin

        ## Flatten the metrics and drop the number of events in each bin
        metrics = run_totals[:,1:].flatten(order='F')
        metrics = np.expand_dims(metrics, 0)

        ## Get the names of the columns and convert the metrics to a dataframe
        mcols = [ met+str(i) for met in met_names for i in range(-1, self.n_bins) ]
        mdf = pd.DataFrame( data=metrics, index=[self.name], columns=mcols )

        ## Expand, label and convert the histogram to a dataframe
        hist1D = np.expand_dims(hist1D, 0)
        hcols = [ 'hist'+str(i) for i in range(self.n_bins) ]
        hdf = pd.DataFrame( data=hist1D, index=[self.name], columns=hcols )

        ## Write the combined dataframe to the csv
        df = pd.concat( [ mdf, hdf, self.get_info() ], axis=1 ) ## Combine the performance dataframe with info on the network
        fnm = Path( self.save_dir, self.name, 'perf.csv' )
        df.to_csv( fnm, mode='w' )

        ## Save the 2D histogram to the output folder
        hnm = Path( self.save_dir, self.name, 'hist2d.png' )
        myPL.save_hist2D( hist2D, hnm, [0, self.bin_max2D, 0, self.bin_max2D ], [[0,self.bin_max2D],[0,self.bin_max2D]] )

        hnm = Path( self.save_dir, self.name, 'outp2d.png' )
        myPL.save_hist2D( outp2D, hnm, [-100, 500, -200, 200 ], [[-100,500],[0,0]] )

        ## Turn the sampler back on for the validation set
        self.valid_loader.dataset.weight_on()

    def get_info(self):
        """ This function is used to return preselected information about the network and
            the training using the class attributes. It is use primarily for the parallel coordinate plot.
            It returns a dataframe.
        """
        columns = [ 'do_rot',
                    'weight_to',
                    'weight_ratio',
                    'weight_shift',
                    'v_frac',
                    'batch_size',
                    'n_train_files',
                    'n_valid_files',
                    'train_size',
                    'valid_size',
                    'act',
                    'depth',
                    'width',
                    'skips',
                    'nrm',
                    'drpt',
                    'loss_fn',
                    'lr',
                    'skn_weight',
                    'epochs_trained',
                    'best_epoch',
                    'cut_calo',
                    'cut_track', ]
        data = [[ self.__dict__[c] for c in columns ]]
        return pd.DataFrame(data=data, index=[self.name], columns=columns)

    def update_dict(self, vars):
        """
        Updates the attribute dictionary with new local scope variables (only numbers or strings).
        Much quicker than simply running self.var = var many times.
        We want local variables to be attributes as the entire dictionary is saved later
        allowing us to quickly store model configuration.
        """
        self.__dict__.update( (k,v) for k,v in vars.items() if k != 'self' )
