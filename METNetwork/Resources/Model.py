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
        stats = T.tensor(np.loadtxt(Path(self.data_dir, 'stats.csv'), skiprows=1, delimiter=','), dtype=T.float32)
        self.net.register_buffer('preproc_stats', stats)

        ## Select the device and move the network
        self.device = myUT.sel_device('auto')
        self.net.to(self.device)
        print(self.net)

    def setup_training(self, opt_nm, lr, reg_loss_nm, dst_loss_nm, dst_weight, grad_clip, b_size, n_workers):
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

        ## Create the pytorch dataloaders (works for both types of datset)
        loader_kwargs = {'batch_size':b_size, 'num_workers':n_workers, 'drop_last':True, 'pin_memory':True}
        self.train_loader = DataLoader(self.train_set, **loader_kwargs)
        self.valid_loader = DataLoader(self.valid_set, **loader_kwargs)

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
            # self._epoch(is_train=True)
            # self._epoch(is_train=False)

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

            ## Move the batch to the network device and break into parts (dont use true_et!)
            inputs, targets, weights = myUT.move_dev(batch, self.device)

            ## Calculate the network output
            outputs = self.net(inputs)

            ## Calculate the weighted batch regression loss
            # y_targ = T.abs(targets.T[1].detach())
            # y_targ /= y_targ.mean()

            # reg_loss = T.tensor(0, device=self.device)
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
            - loss.csv -> Recorded loss history of the training performance
            - loss.png -> (Several) plots of the recorded history
        - When the network validation loss improves (below is handled by the save_perf method)
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
            f.write('\n\n'+str(self.net))
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
        at profiles using bins of True ETmiss. This is done both for Tight and for the Network.
        The binned variables are:
            - Resolution (x, y)
            - Deviation from linearity
            - Angular Resolution
        These profiles are combined with information from the class dict to create a performance csv.
        This performance csv only has two lines: the column names and the values. This is so it can be combined
        with results from other networks!
        We also create several a 1D and 2D histograms
            - 1D histogram of the reconstructed and true et (post-processed)
            - 2D histogram of the reconstructed and true x,y (raw)
        """
        print('\nImprovement detected! Saving additional information')

        ## The bin setup to use for the profiles
        n_bins = 40
        m_bins = 400
        bins = np.linspace(0, m_bins, n_bins+1)
        bins2d = [ np.linspace(-2, 4, n_bins+1), np.linspace(-3, 3, n_bins+1) ]

        targ_means = self.net.preproc_stats[0,-2:] ## The stats needed to unormalise the target space
        targ_sdevs = self.net.preproc_stats[1,-2:]

        ## The performance metrics to be calculated are stored in a running total matrix
        met_names = [ 'Res', 'Lin', 'Ang' ]
        r_idx = [1, 3] ## Which columns are root means as opposed to means (+1)
        binned_totals = np.zeros((n_bins+1, len(met_names)+1), dtype=np.float64) ## rows = (total+bin1, bin2...) x cols = (n_events + *met_names)

        ## Build a 1D and 2D histogram of the output magnitude
        h_net_et = np.zeros(n_bins)
        h_tru_et = np.zeros(n_bins)
        h_net_xy = np.zeros((n_bins, n_bins))
        h_tru_xy = np.zeros((n_bins, n_bins))

        ## Configure pytorch, the network and the loader appropriately
        T.set_grad_enabled(False)
        self.net.eval()
        self.valid_loader.dataset.weight_off()

        all_outputs = []
        all_targets = []
        ## Iterate through the validation set
        for batch in tqdm(self.valid_loader, desc='perfm', ncols=80, ascii=True):

            ## Get the network outputs
            inputs, targets = myUT.move_dev(batch[:-1], self.device)
            outputs = self.net(inputs)

            all_outputs.append
            all_targets.append

            ## Undo the normalisation on the outputs and the targets
            net_xy = (outputs * targ_sdevs + targ_means) / 1000
            tru_xy = (targets * targ_sdevs + targ_means) / 1000
            net_et = T.norm(net_xy, dim=1)
            tru_et = T.norm(tru_xy, dim=1)

            ## Calculate the performance metrics
            nev = T.ones_like(tru_et) ## For keeping track of how many events per bin
            res = ((net_xy - tru_xy)**2).mean(dim=1)
            lin = (net_et - tru_et) / (tru_et + 1e-8)
            ang = T.acos( T.sum(net_xy*tru_xy, dim=1) / (net_et*tru_et+1e-8) )**2 ## Calculated using the dot product

            ## Combine the performance metrics into a single pandas dataframe
            combined = T.vstack([nev, res, lin, ang]).T
            df = pd.DataFrame(myUT.to_np(combined), columns=['n_events']+met_names)

            ## Add bins in True_ET, we do this manually so we can deal with overflow! No np.digitize!
            df['True_ET_bins'] = myUT.to_np(T.clamp(tru_et*n_bins/m_bins, 0, n_bins-1).int())

            ## Now we make the profiles using groupby, and add to the running totals using the indices
            groupby = df.groupby('True_ET_bins', as_index=False).sum().to_numpy()
            binned_totals[groupby[:, 0].astype(int)+1] += groupby[:, 1:]

            ## We update the histograms
            h_net_et += np.histogram(myUT.to_np(net_et), bins=bins)[0]
            h_tru_et += np.histogram(myUT.to_np(tru_et), bins=bins)[0]
            h_net_xy += np.histogram2d(*myUT.to_np(outputs).T[[0,1]], bins=bins2d)[0]
            h_tru_xy += np.histogram2d(*myUT.to_np(targets).T[[0,1]], bins=bins2d)[0]

        ## Get the set total and turn into means (and root means)
        binned_totals[0] = binned_totals.sum(axis=0)
        binned_totals = binned_totals / (binned_totals[:,0:1]+1e-8)
        binned_totals[:, r_idx] = np.sqrt(binned_totals[:, r_idx])

        ## Flatten the metrics, drop the number of events, create names and make a dataframe
        metrics = binned_totals[:,1:].flatten(order='F')
        mcols = [ met+str(i) for met in met_names for i in range(-1, n_bins) ]
        metric_df = pd.DataFrame( data=[metrics], index=[self.name], columns=mcols )

        ## Get a dataframe from the class dict
        dict_df = pd.DataFrame.from_dict([self.get_dict()]).set_index('name')

        ## Write the combined dataframe to the csv
        out_df = pd.concat([metric_df, dict_df], axis=1)
        out_df.to_csv(Path(self.save_dir, self.name, 'perf.csv'), mode='w')

        ## Normalise the magnitude histograms and save them to file
        mid_bins =(bins[1:]+bins[:-1])/2
        h_net_et /= np.sum(h_net_et)
        h_tru_et /= np.sum(h_tru_et)
        np.savetxt(Path(self.save_dir, self.name, 'hist_et.csv'), np.vstack([mid_bins, h_net_et, h_tru_et]).T)
        fig = plt.figure( figsize=(8,4) )
        ax = fig.add_subplot(111)
        ax.plot(h_net_et)
        ax.plot(h_tru_et)
        fig.savefig(Path(self.save_dir, self.name, 'mplot.png'))
        plt.close(fig)

        ## Normalise the 2d histograms
        h_net_xy /= np.sum(h_net_xy)
        h_tru_xy /= np.sum(h_tru_xy)

        fig = plt.figure( figsize=(8,4) )
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(h_net_xy.T)
        ax2.imshow(h_tru_xy.T)
        fig.savefig(Path(self.save_dir, self.name, 'plot.png'))
        plt.show()
        # plt.close(fig)

        test = myPL.projectiontion2D_ndarray(x, y, z)
        print(test)

        exit()
        # plt.show()
        # exit()

        ## Save the 2D histogram to the output folder
        # hnm = Path( self.save_dir, self.name, 'hist2d.png' )
        # myPL.save_hist2D( hist2D, hnm, [0, self.bin_max2D, 0, self.bin_max2D ], [[0,self.bin_max2D],[0,self.bin_max2D]] )

        # hnm = Path( self.save_dir, self.name, 'outp2d.png' )
        # myPL.save_hist2D( outp2D, hnm, [-100, 500, -200, 200 ], [[-100,500],[0,0]] )

    def get_dict(self):
        """
        Creates a dictionary using all strings, intergers and floats from the state dict.
        This should be sufficient for recording all hyperparameters for the network.
        """
        return { k:str(v) for k, v in self.__dict__.items() if isinstance(v, (str, int, float)) }

    def update_dict(self, vars):
        """
        Updates the attribute dictionary with new local scope variables (only numbers or strings).
        Much quicker than simply running self.var = var many times.
        We want local variables to be attributes as the entire dictionary is saved later
        allowing us to quickly store model configuration.
        """
        self.__dict__.update( (k,v) for k,v in vars.items() if k != 'self' )
