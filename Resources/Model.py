from Resources import Plotting as myPL
from Resources import Networks as myNN
from Resources import Datasets as myDS

import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from itertools import count
from collections import Counter

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class METNET_Agent(object):
    def __init__(self, name, save_dir ):
        self.save_dir = save_dir
        self.name = name

        ## How to setup the bins for performance metrics
        self.n_bins = 10
        self.bin_max = 400

    def setup_dataset( self, data_dir, do_rot, valid_frac, n_ofiles, chnk_size, batch_size, n_workers ):
        """ Uses a pytorch dataloader to get the training and validation sets
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        ## The data dir and the stat file depend on if we are doing rotations
        self.data_dir  = Path( self.data_dir, "Rotated" if do_rot else "Raw" )
        self.stat_file = Path( self.data_dir, "stats.csv" )

        ## Build the list of files that will be used in the train and validation set
        train_files, valid_files = myDS.buildTrainAndValidation( self.data_dir, valid_frac )
        self.n_train_files = len(train_files)
        self.n_valid_files = len(valid_files)

        ## Get the defined iterable dataset objects
        train_set = myDS.METDataset( train_files, n_ofiles, chnk_size )
        valid_set = myDS.METDataset( valid_files, n_ofiles, chnk_size )

        ## Create the pytorch dataloaders with num_workers accounting for low numbers of files
        self.train_loader = DataLoader( train_set, batch_size=batch_size, drop_last=True, num_workers=min(self.n_train_files, n_workers) )
        self.valid_loader = DataLoader( valid_set, batch_size=batch_size, drop_last=True, num_workers=min(self.n_valid_files, n_workers) )

        ## Report on the number of files/samples used
        self.train_size = len(train_set)
        self.valid_size = len(valid_set)
        print( "Train set:       {} samples (in {} files)".format( self.train_size, self.n_train_files ) )
        print( "Validation set:  {} samples (in {} files)".format( self.valid_size, self.n_valid_files ) )

    def setup_network( self, act, depth, width, skips, nrm, drpt ):
        """ This initialises the mlp network structure
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        ## We get the number of inputs based on the pre-process
        n_in = 74 if self.do_rot else 76

        ## Creating the neural network
        self.network = myNN.MET_MLP( "res_mlp", n_in, act, depth, width, skips, nrm, drpt )

    def setup_training( self, loss_fn, lr, clip_grad ):
        """ Sets up variables used for training, including the optimiser
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        ## Initialise the optimiser
        self.optimiser = optim.Adam( self.network.parameters(), lr=lr )

        self.trn_hist = []
        self.vld_hist = []
        self.tim_hist = []
        self.epochs_trained = 0

        self.loss_plot = myPL.loss_plot( self.name )

    def _train_epoch( self ):
        """ This function performs one epoch of training on data provided by the train_loader
            It will also update the graphs after a certain number of batches pass
        """
        ## Put the nework into training mode (for batch_norm/droput)
        self.network.train()
        running_loss = 0

        for i, (inputs, targets) in enumerate( tqdm( self.train_loader, desc="Training", ncols=80, unit="" ) ):

            ## Zero out the gradients
            self.optimiser.zero_grad()

            ## Move data to the network device (GPU)
            inputs = inputs.to(self.network.device)
            targets = targets.to(self.network.device)

            ## Calculate the network output
            outputs = self.network( inputs )

            ## Calculate the batch loss
            loss = self.loss_fn( outputs, targets )

            ## Perform gradient descent
            loss.backward()
            if self.clip_grad > 0: nn.utils.clip_grad_value_( self.network.parameters(), self.clip_grad )
            self.optimiser.step()

            ## Update the running loss
            running_loss += loss.item()

        ## Update loss history and epoch counter
        self.trn_hist.append( running_loss / (i+1) ) ## Divide by i as we dont include final batches
        self.epochs_trained += 1

    def _test_epoch(self):
        """ This function performs one epoch of testing on data provided by the validation loader
        """

        with T.no_grad():
            self.network.eval()
            running_loss = 0
            for i, (inputs, targets) in enumerate( tqdm( self.valid_loader, desc="Testing ", ncols=80, unit="" ) ):
                inputs = inputs.to(self.network.device)
                targets = targets.to(self.network.device)
                outputs = self.network( inputs )
                loss = self.loss_fn( outputs, targets )
                running_loss += loss.item()
            self.vld_hist.append( running_loss / (i+1) ) ## Divide by i as we dont include final batches

    def run_training_loop( self, max_epochs = 10, patience = 20, sv_every = 20 ):
        """ This is the main loop which cycles epochs of train and test
            It also updates graphs and saves the network
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        for epoch in range(1, max_epochs+1):

            print( "\nEpoch: {}".format(epoch) )

            ## For time keeping
            e_start_time = time.time()

            ## Run the test/train cycle
            self._test_epoch()
            self._train_epoch()

            ## Shuffle the file order in the datasets
            self.train_loader.dataset.shuffle_files()
            self.valid_loader.dataset.shuffle_files()

            ## For time keeping
            self.tim_hist.append( time.time() - e_start_time )

            ## Printing loss
            print( "Average loss on training set:   {:.5}".format( self.trn_hist[-1] ) )
            print( "Average loss on validaiton set: {:.5}".format( self.vld_hist[-1] ) )
            print( "Epoch Time: {:.5}".format( self.tim_hist[-1] ) )

            ## Calculate the number of bad epochs
            self.best_epoch = np.argmin(self.vld_hist) + 1
            self.bad_epochs = len(self.vld_hist) - self.best_epoch

            ## At the end of every epoch we save something, even if it is just logging
            self.save()

            ## We check if we have exceeded the patience
            if self.bad_epochs > 0:
                print( "Bad Epoch Number: {:}".format( bad_epochs ) )
                if self.bad_epochs > patience:
                    print( "Patience Exceeded: Stopping training!" )
                    return 0

        print( "\nMax number of epochs completed!" )
        return 0

    def save( self, flag ):
        """
        Creates a save directory that includes the following
            - models   -> A folder containing the network and optimiser versions
            - info.txt -> A file containing the network setup and description
            - loss.csv -> Hitory of the train + validation losses (with png)
            - stat.csv -> Values used to normalise the data for the network
        """

        ## We work out the flag for saving, either "best", "chkpnt", "none"
        if self.bad_epochs==0:
            flag = "best"
        elif self.epochs_trained%self.sv_every==0:
            flag = str(self.epochs_trained)
        else:
            flag = "none"

        ## The full name of the save directory
        full_name = Path( self.save_dir, self.name )

        ## Our first ever call to this function should remove the contents of the directory and recreate
        if self.epochs_trained == 1:
            if full_name.exists():
                print("Deleting files!")
                shutil.rmtree(full_name)
        full_name.mkdir(parents=True, exist_ok=True)

        ## Save the network and optimiser tensors: Only for "best" and "chkpnt"!
        if flag != "none":
            model_folder = Path( full_name, "models" )
            model_folder.mkdir(parents=True, exist_ok=True)
            T.save( self.network.state_dict(),   Path( model_folder, "net_"+flag ) )
            T.save( self.optimiser.state_dict(), Path( model_folder, "opt_"+flag ) )

        ## Save a file containing the network setup and description (based on class dict): All flags!
        with open( Path( full_name, "info.txt" ), 'w' ) as f:
            for key in self.__dict__:
                attr = self.__dict__[key]                   ## Information here is more inclusive than get_info below
                if isinstance(attr, list): attr = min(attr) ## Lists are a pain to save, so we just use min
                if len(str(attr)) > 50: continue            ## Long strings are ignored, these are pointers and class types
                f.write( "{:15} = {}\n".format(key, str(attr)) )
            f.write( "\n\n"+str(self.network) )
            f.write( "\n\n"+str(self.optimiser) )
            f.write( "\n" ) ## One line for whitespace

        ## Save the loss history, with epoch times, and a png of the graph: All flags!
        hist_array = np.transpose( np.vstack(( self.trn_hist, self.vld_hist, self.tim_hist )) )
        np.savetxt( Path( full_name, "train_hist.csv" ), hist_array )
        self.loss_plot.save( self.trn_hist, self.vld_hist, Path( full_name, "loss.png" ) )

        ## Save a copy of the stat file in the network directory: Only on the first iteration!
        if self.epochs_trained==1:
            shutil.copyfile( self.stat_file, Path( full_name, "stat.csv") )

    def load( self, flag ):

        ## Get the name of the directories
        full_name = Path( self.save_dir, self.name )
        model_folder = Path( full_name, "models" )

        ## Load the network and optimiser
        self.network.load_state_dict(   T.load( Path( model_folder, "net_"+flag ) ) )
        self.optimiser.load_state_dict( T.load( Path( model_folder, "opt_"+flag ) ) )

        ## Load the train history
        previous_data = np.loadtxt( Path( full_name, "train_hist.csv" ) )
        self.trn_hist = previous_data[:,0].tolist()
        self.vld_hist = previous_data[:,1].tolist()
        self.tim_hist = previous_data[:,2].tolist()
        self.epochs_trained = len(self.trn_hist)

    def save_best_perf( self ):
        print("\nSaving additional information and performance using best network")

        ## If we are saving the best version of the network, we must make sure that it is loaded!
        model_name = Path( self.save_dir, self.name, "models", "net_best" )
        self.network.load_state_dict( T.load( model_name ) )

        ## Calculating performance will require un-normalising our outputs, so we need the stats
        stats = np.loadtxt(self.stat_file, skiprows=1, delimiter=",")
        means = T.from_numpy(stats[0,-2:]).to(self.network.device) ## Just need the target (True EX, EY) stats
        devs  = T.from_numpy(stats[1,-2:]).to(self.network.device)

        ## The performance metrics to be calculated are stored in a running total matrix
        run_totals = np.zeros( (self.n_bins+1, 5) ) ## Nbins (incl totals), Loss, Res, Ang, DLin

        ## Iterate through the validation set
        with T.no_grad():
            self.network.eval()
            for (inputs, targets) in tqdm( self.valid_loader, desc="Performance", ncols=80, unit="" ):

                ## Pass the information through the network
                inputs = inputs.to(self.network.device)
                targets = targets.to(self.network.device)
                outputs = self.network( inputs )

                ## Un-normalise the outputs and the targets
                real_outp = ( outputs*devs + means ) / 1000 ## Convert to GeV (Dont like big numbers)
                real_targ = ( targets*devs + means ) / 1000

                ## Calculate the magnitudes of the vectors
                targ_mag = T.norm(real_targ, dim=1)
                outp_mag = T.norm(real_outp, dim=1)

                ## Get the bin numbers from the true met magnitude
                bins = T.clamp( targ_mag // (self.bin_max/self.n_bins), 0, self.n_bins-1 ).int().cpu().numpy()

                ## Calculate the batch totals
                batch_totals = T.zeros( ( len(inputs), 5 ) )
                dot = T.sum( real_outp * real_targ, dim=1 ) / ( outp_mag * targ_mag + 1e-4 )

                batch_totals[:, 0] = T.ones_like(targ_mag)                                            ## Ones for counting bins
                batch_totals[:, 1] = F.smooth_l1_loss(outputs, targets, reduction="none").sum(axis=1) ## Loss
                batch_totals[:, 2] = ( outp_mag - targ_mag )**2                                       ## Resolution
                batch_totals[:, 3] = T.acos( dot )**2                                                 ## Angular resolution
                batch_totals[:, 4] = ( outp_mag - targ_mag ) / ( targ_mag + 1e-4 )                    ## Deviation from Linearity

                ## Fill in running data by, summing over each bin, bin 0 is reserved for totals
                for b in range(self.n_bins):
                    run_totals[b+1] += batch_totals[ bins==b ].sum(axis=0).cpu().numpy()

        ## Include the totals over the whole dataset by summing and placing it in the first location
        run_totals[0] = run_totals.sum(axis=0, keepdims=True)
        run_totals[:,0] = np.clip( run_totals[:,0], 1, None ) ## Just incase some of the bins were empty, dont wana divide by 0

        ## Turn the totals into means or RMSE values
        run_totals[:,1] = run_totals[:,1] / run_totals[:,0]
        run_totals[:,2] = np.sqrt( run_totals[:,2] / run_totals[:,0] )
        run_totals[:,3] = np.sqrt( run_totals[:,3] / run_totals[:,0] )
        run_totals[:,4] = run_totals[:,4] / run_totals[:,0]

        ## Flatten the metrics and drop the number of events in each bin
        metrics = run_totals[:,1:].flatten(order='F')
        metrics = np.expand_dims(metrics, 0)

        ## Getting the names of the columns
        cols = []
        for met in [ "Loss", "Res", "Ang", "DLin" ]:
            cols += [ met+str(i) for i in range(-1, self.n_bins) ]

        ## Write the dataframe to the csv
        fnm = Path( self.save_dir, self.name, "perf.csv" )
        df = pd.DataFrame( data=metrics, index=[self.name], columns=cols )
        df = pd.concat( [ df, self.get_info() ], axis=1 ) ## Combine the performance dataframe with info on the network
        df.to_csv( fnm, mode="w" )

    def get_info(self):
        """ This function is used to return preselected information about the network and
            the training using the class attributes. It returns a dataframe.
        """
        columns = [ "do_rot",
                    "valid_frac",
                    "batch_size",
                    "n_train_files",
                    "n_valid_files",
                    "train_size",
                    "valid_size",
                    "act",
                    "depth",
                    "width",
                    "skips",
                    "nrm",
                    "drpt",
                    "loss_fn",
                    "lr",
                    "clip_grad",
                    "epochs_trained",
                    "best_epoch" ]
        data = [[ self.__dict__[c] for c in columns ]]
        return pd.DataFrame(data=data, index=[self.name], columns=columns)
