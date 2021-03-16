from Resources import Plotting as myPL
from Resources import Networks as myNN
from Resources import Datasets as myDS

import time
import glob
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

        self.n_bins = 10            ## How many bins to use for the performance metrics

    def setup_dataset( self, data_dir, stat_file, valid_frac, n_ofiles, chnk_size, batch_size, n_workers ):
        """ Uses a pytorch dataloader to get the training and validation sets
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        ## Load the stat file and the x indeces for pre-processing
        self.stat_file = stat_file

        ## Build the list of files that will be used in the train and validation set
        train_files, valid_files = myDS.buildTrainAndValidation( data_dir, valid_frac )
        self.n_train_files = len(train_files)
        self.n_valid_files = len(valid_files)

        ## Get the defined iterable dataset objects
        train_set = myDS.METDataset( train_files, n_ofiles, chnk_size )
        valid_set = myDS.METDataset( valid_files, n_ofiles, chnk_size )

        ## Create the pytorch dataloaders with num_workers accounting for low numbers of files
        self.train_loader = DataLoader( train_set, batch_size = batch_size, num_workers = min(self.n_train_files, n_workers) )
        self.valid_loader = DataLoader( valid_set, batch_size = batch_size, num_workers = min(self.n_valid_files, n_workers) )

        ## Report on the number of files/samples used
        self.train_size = len(train_set)
        self.valid_size = len(valid_set)
        print( "Train set:       {} samples (in {} files)".format( self.train_size, self.n_train_files ) )
        print( "Validation set:  {} samples (in {} files)".format( self.valid_size, self.n_valid_files ) )

    def setup_network( self, n_in, act, depth, width, skips, nrm, drpt ):
        """ This initialises the mlp network structure
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

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

        for (inputs, targets) in tqdm( self.train_loader, desc="Training", ncols=80, unit="" ):

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
        self.trn_hist.append( running_loss / len(self.train_loader) )
        self.epochs_trained += 1

    def _test_epoch(self):
        """ This function performs one epoch of testing on data provided by the validation loader
        """

        with T.no_grad():
            self.network.eval()
            running_loss = 0
            for (inputs, targets) in tqdm( self.valid_loader, desc="Testing ", ncols=80, unit="" ):
                inputs = inputs.to(self.network.device)
                targets = targets.to(self.network.device)
                outputs = self.network( inputs )
                loss = self.loss_fn( outputs, targets )
                running_loss += loss.item()
            self.vld_hist.append( running_loss / len(self.valid_loader) )

    def run_training_loop( self, patience = 20, sv_every = 20 ):
        """ This is the main loop which cycles epochs of train and test
            It also updates graphs and saves the network
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        for epoch in count(1):

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

            ## Saving a network checkpoint
            if self.epochs_trained%sv_every == 0:
                self.save( str(self.epochs_trained) )

            ## Calculate the number of bad epochs
            self.best_epoch = np.argmin(self.vld_hist) + 1
            bad_epochs = len(self.vld_hist) - self.best_epoch

            ## Checking bad epochs and either saving or testing patience
            if bad_epochs == 0:
                self.save( "best" )
            else:
                print( "Bad Epoch Number: {:}".format( bad_epochs ) )
                if bad_epochs > patience:
                    print( "Patience Exceeded: Stopping training!" )
                    return 0

    def save_perf( self, flag ):
        print("\nCalculating additional performance metrics")

        ## If we are saving the best version of the network, we must make sure that it is loaded!
        if flag=="best":
            model_name = Path( self.save_dir, self.name, "models", "net_best" )
            self.network.load_state_dict( T.load( model_name ) )

        ## Calculating performance will require un-normalising our outputs, so we need the stats
        stats = np.loadtxt(self.stat_file, skiprows=1, delimiter=",")
        means = T.from_numpy(stats[0,-2:]).to(self.network.device)
        devs  = T.from_numpy(stats[1,-2:]).to(self.network.device)

        ## The performance metrics to be calculated are stored in a running total matrix
        run_totals = np.zeros( (self.n_bins, 4) ) ## Nbins, res, ang, lin

        ## Iterate through the validation set
        with T.no_grad():
            self.network.eval()
            for (inputs, targets) in tqdm( self.valid_loader, desc="Performance", ncols=80, unit="" ):

                ## Pass the information through the network
                inputs = inputs.to(self.network.device)
                targets = targets.to(self.network.device)
                outputs = self.network( inputs )

                ## Un-normalise the outputs and the targets
                targets = ( targets*devs + means ) / 1000 ## Convert to GeV (Dont like big numbers)
                outputs = ( outputs*devs + means ) / 1000

                ## Calculate the magnitudes of the vectors
                targ_mag = T.norm(targets, dim=1)
                outp_mag = T.norm(outputs, dim=1)

                ## Get the bin numbers from the true met magnitude
                bins = T.clamp( targ_mag // (300/self.n_bins), 0, self.n_bins-1 ).int().cpu().numpy()

                ## Calculate the batch values for resolution, ang res and linearity
                dot = T.sum( outputs * targets, dim=1 ) / ( outp_mag * targ_mag + 1e-4 )
                res_tmp = ( outp_mag - targ_mag )**2
                ang_tmp = T.acos( dot )**2
                lin_tmp = ( outp_mag - targ_mag ) / ( targ_mag + 1e-4 )

                ## Combine the metrics into a single numpy array
                stack = T.stack( [res_tmp, ang_tmp, lin_tmp], -1).cpu().numpy() ## shape = batch x 3

                ## Fill in out running data using the metric array and the bin number
                for b in range(self.n_bins):
                    chunk = stack[ bins==b ]
                    new_info = np.concatenate( [ [len(chunk)], chunk.sum(axis=0) ] ) ## Len chunks will give the number of elements in the bin
                    run_totals[b] += new_info

        ## Getting the performance on the whole dataset
        inclusive = np.concatenate( [run_totals.sum(axis=0, keepdims=True), run_totals ], axis=0 )
        res = np.sqrt( inclusive[:,1] / inclusive[:,0] )
        ang = np.sqrt( inclusive[:,2] / inclusive[:,0] )
        lin = inclusive[:,3] / inclusive[:,0]

        ## Getting the names of the columns
        cols = [ "loss" ] \
             + [ "res"+str(i) for i in range(-1, self.n_bins) ] \
             + [ "ang"+str(i) for i in range(-1, self.n_bins) ] \
             + [ "lin"+str(i) for i in range(-1, self.n_bins) ]
        perm = np.concatenate( [ [self.vld_hist[-1]], res, ang, lin ] )
        perm = np.expand_dims(perm, 0)

        ## Values needed for writing to the csv file
        fnm = Path( self.save_dir, self.name, "perf.csv" )
        doh = not fnm.is_file()
        wrt = "a" if fnm.is_file() else "w"
        idx = [self.name + "_" + flag]

        ## Write the dataframe to the csv
        df = pd.DataFrame(data=perm, index=idx, columns=cols)
        df.to_csv( fnm, mode=wrt, header=doh )

        ## Now that we have written to the performance we want to update, not overwrite
        self.written_perf = True

    def save( self, flag ):
        """
        Creates a save directory that includes the following
            - models   -> A folder containing the network and optimiser versions
            - info.txt -> A file containing the network setup and description
            - loss.csv -> Hitory of the train + validation losses (with png)
            - perf.csv -> A csv containing the network performances at various stages
            - stat.csv -> Values used to normalise the data for the network
        """

        ## The full name of the save directory
        full_name = Path( self.save_dir, self.name )

        ## Our first ever call to this function should remove the contents of the directory and recreate
        if self.epochs_trained==1: shutil.rmtree(full_name)
        full_name.mkdir(parents=True, exist_ok=True)

        ## Save the network and optimiser tensors
        model_folder = Path( full_name, "models" )
        model_folder.mkdir(parents=True, exist_ok=True)
        T.save( self.network.state_dict(),   Path( model_folder, "net_"+flag ) )
        T.save( self.optimiser.state_dict(), Path( model_folder, "opt_"+flag ) )

        ## Save a file containing the network setup and description (based on class dict)
        with open( Path( full_name, "info.txt" ), 'w' ) as f:
            for key in self.__dict__:
                attr = self.__dict__[key]
                if isinstance(attr, list): attr = min(attr)
                str_ver = str(attr)
                if len(str_ver) > 50:
                    continue
                f.write( "{:15} = {}\n".format(key, str(attr)) )
            f.write( "\n\n"+str(self.network) )
            f.write( "\n\n"+str(self.optimiser) )

        ## Save the loss history, with epoch times, and a png of the graph
        hist_array = np.transpose( np.vstack(( self.trn_hist, self.vld_hist, self.tim_hist )) )
        np.savetxt( Path( full_name, "train_hist.csv" ), hist_array )
        self.loss_plot.save( self.trn_hist, self.vld_hist, Path( full_name, "loss.png" ) )

        ## Save the performance of the network (not for the best! that happens at end of training loop!)
        if flag != "best":
            self.save_perf(flag)

        ## Save a copy of the stat file in the network directory (only on the first iteration)
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
