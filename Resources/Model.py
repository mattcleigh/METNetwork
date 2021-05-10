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
from torch.utils.data import DataLoader, RandomSampler

class METNET_Agent(object):
    def __init__(self, name, save_dir ):
        self.save_dir = save_dir
        self.name = name

        ## How to setup the bins for performance metrics
        self.n_bins = 50
        self.bin_max = 400

    def setup_dataset( self, data_dir, do_rot,
                             weight_to, weight_ratio, weight_shift,
                             v_frac, n_ofiles, chnk_size,
                             batch_size, n_workers ):
        """ Uses a pytorch dataloader to get the training and validation sets
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        ## The data dir, stat and the histogram file depend on if we are doing rotations
        self.data_dir  = Path( self.data_dir, "Rotated" if do_rot else "Raw" )
        self.stat_file = Path( self.data_dir, "stats.csv" )
        hist_file = Path( self.data_dir, "hist.csv" )

        ## Build the list of files that will be used in the train and validation set
        train_files, valid_files = myDS.buildTrainAndValidation( self.data_dir, v_frac )
        self.n_train_files = len(train_files)
        self.n_valid_files = len(valid_files)

        ## Get the iterable dataset objects
        train_set = myDS.StreamMETDataset( train_files, n_ofiles, chnk_size, hist_file, weight_to, weight_ratio, weight_shift )
        valid_set = myDS.StreamMETDataset( valid_files, n_ofiles, chnk_size, hist_file, weight_to, weight_ratio, weight_shift )

        ## Create the pytorch dataloaders (works for both types of datset)
        self.train_loader = DataLoader( train_set,
                                        drop_last   = True,                                ## Causes errors with batch norm if batch = 1
                                        batch_size  = batch_size,                          ## Set by training script
                                        sampler     = train_set.sampler,                   ## Is always None for iterable dataset
                                        num_workers = min(self.n_train_files, n_workers),  ## Dont want dead workers if not enough files
                                        pin_memory  = True )                               ## Improves CPU->GPU transfer times
        self.valid_loader = DataLoader( valid_set,
                                        drop_last   = True,
                                        batch_size  = batch_size,
                                        sampler     = valid_set.sampler,
                                        num_workers = min(self.n_valid_files, n_workers),
                                        pin_memory  = True )

        ## Report on the number of files/samples used
        self.train_size = len(train_set)
        self.valid_size = len(valid_set)
        print( "Train set:       {:10} samples in {:4} files".format( self.train_size, self.n_train_files ) )
        print( "Validation set:  {:10} samples in {:4} files".format( self.valid_size, self.n_valid_files ) )

    def setup_network( self, act, depth, width, skips, nrm, drpt ):
        """ This initialises the mlp network structure
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        ## We get the number of inputs based on the pre-process
        n_in = 74 if self.do_rot else 76

        ## Creating the neural network
        self.network = myNN.MET_MLP( "res_mlp", n_in, act, depth, width, skips, nrm, drpt )

    def setup_training( self, loss_fn, lr ):
        """ Sets up variables used for training, including the optimiser
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        ## Initialise the optimiser
        self.optimiser = optim.Adam( self.network.parameters(), lr=lr )

        ## The history of train losses, epoch times etc ...
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

        for i, (inputs, targets, weights) in enumerate( tqdm( self.train_loader, desc="Training", ncols=80 ) ):

            ## Zero out the gradients
            self.optimiser.zero_grad()

            ## Move data to the network device (GPU)
            inputs = inputs.to(self.network.device)
            targets = targets.to(self.network.device)
            weights = weights.to(self.network.device)

            ## Calculate the network output
            outputs = self.network( inputs )

            ## Calculate the batch loss
            loss = ( self.loss_fn( outputs, targets ).mean( dim=1 ) * weights ).mean()

            ## Calculate the gradients
            loss.backward()

            ## Update the parameters
            self.optimiser.step()

            ## Track the running loss
            running_loss += loss.item()

        ## Update loss history and epoch counter
        self.trn_hist.append( running_loss / (i+1) ) ## Divide by i as we dont include final (unfilled) batches
        self.epochs_trained += 1

    def _test_epoch(self):
        """ This function performs one epoch of testing on data provided by the validation loader
        """

        with T.no_grad():
            self.network.eval()
            running_loss = 0
            for i, (inputs, targets, weights) in enumerate( tqdm( self.valid_loader, desc="Testing ", ncols=80 ) ):
                inputs = inputs.to(self.network.device)
                targets = targets.to(self.network.device)
                weights = weights.to(self.network.device)
                outputs = self.network( inputs )
                loss = ( self.loss_fn( outputs, targets ).mean( dim=1 ) * weights ).mean()
                running_loss += loss.item()
            self.vld_hist.append( running_loss / (i+1) ) ## Divide by i as we dont include final (unfilled) batches

    def run_training_loop( self, max_epochs = 10, patience = 20, sv_every = 20 ):
        """ This is the main loop which cycles epochs of train and test
            It saves the network and checks for early stopping
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        for epoch in range(1, max_epochs+1):

            print( "\nEpoch: {}".format(epoch) )

            ## For time keeping
            e_start_time = time.time()

            ## Run the test/train cycle
            self._train_epoch()
            self._test_epoch()

            ## Shuffle the file order in the datasets (does nothing for map-style datasets)
            self.train_loader.dataset.shuffle_files()
            self.valid_loader.dataset.shuffle_files()

            ## For time keeping
            self.tim_hist.append( time.time() - e_start_time )

            ## Printing loss
            print( "Average loss on training set:   {:.5}".format( self.trn_hist[-1] ) )
            print( "Average loss on validaiton set: {:.5}".format( self.vld_hist[-1] ) )
            print( "Epoch Time: {:.5}".format( self.tim_hist[-1] ) )

            ## Calculate the best epoch and the number of bad epochs
            self.best_epoch = np.argmin(self.vld_hist) + 1
            self.bad_epochs = len(self.vld_hist) - self.best_epoch

            ## At the end of every epoch we save something, even if it is just logging
            self.save()

            ## If the validation loss did not decrease, we check if we have exceeded the patience
            if self.bad_epochs > 0:
                print( "Bad Epoch Number: {:}".format( self.bad_epochs ) )
                if self.bad_epochs > patience:
                    print( "Patience Exceeded: Stopping training!" )
                    return 0

        print( "\nMax number of epochs completed!" )
        return 0

    def save( self ):
        """
        Creates a save directory that includes the following
            - models   -> A folder containing the network and optimiser versions
            - info.txt -> A file containing the network setup and description
            - loss.csv -> Hitory of the train + validation losses (with png)
            - stat.csv -> Values used to normalise the data for the network
            - perf.csv -> Performance metrics on the validation set for the best network
        """

        ## We work out the flags for saving the pytorch model files, "latest", "best" and/or "chkpnt"
        model_flags = [ "latest" ]
        if self.bad_epochs==0: model_flags.append( "best" )
        if self.epochs_trained%self.sv_every==0: model_flags.append( str(self.epochs_trained) )

        ## The full name of the save directory
        full_name = Path( self.save_dir, self.name )

        ## Our first ever call to this function should remove the contents of the directory and recreate
        if self.epochs_trained == 1:
            if full_name.exists():
                print("Deleting files!")
                shutil.rmtree(full_name)
        full_name.mkdir(parents=True, exist_ok=True)

        ## Save the network and optimiser tensors for the latest, best and checkpoint versions
        for flag in model_flags:
            model_folder = Path( full_name, "models" )
            model_folder.mkdir(parents=True, exist_ok=True)
            T.save( self.network.state_dict(),   Path( model_folder, "net_"+flag ) )
            T.save( self.optimiser.state_dict(), Path( model_folder, "opt_"+flag ) )

        ## Save a file containing the network setup and description (based on class dict)
        with open( Path( full_name, "info.txt" ), 'w' ) as f:
            for key in self.__dict__:
                attr = self.__dict__[key]                   ## Information here is more inclusive than get_info below
                if isinstance(attr, list): attr = min(attr) ## Lists are a pain to save, so we just use min
                if len(str(attr)) > 50: continue            ## Long strings are ignored, these are pointers and class types
                f.write( "{:15} = {}\n".format(key, str(attr)) )
            f.write( "\n\n"+str(self.network) )
            f.write( "\n\n"+str(self.optimiser) )
            f.write( "\n" ) ## One line for whitespace

        ## Save the loss history, epoch times, and a png of the graph
        hist_array = np.transpose( np.vstack(( self.trn_hist, self.vld_hist, self.tim_hist )) )
        np.savetxt( Path( full_name, "train_hist.csv" ), hist_array )
        self.loss_plot.save( self.trn_hist, self.vld_hist, Path( full_name, "loss.png" ) )

        ## Save a copy of the stat file in the network directory: Only on the first iteration!
        if self.epochs_trained==1:
            shutil.copyfile( self.stat_file, Path( full_name, "stat.csv") )

        ## If the network is the best, then we also save the performance file
        if self.bad_epochs==0:
            print("\nImprovement detected! Saving additional information")
            self.save_perf()

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
        self.best_epoch = np.argmin(self.vld_hist) + 1
        self.epochs_trained = len(self.trn_hist)

    def save_perf( self ):

        ## Calculating performance will require un-normalising our outputs, so we need the stats
        stats = np.loadtxt(self.stat_file, skiprows=1, delimiter=",")
        means = T.from_numpy(stats[0,-2:]).to(self.network.device) ## Just need the target (True EX, EY) stats
        devs  = T.from_numpy(stats[1,-2:]).to(self.network.device)

        ## The performance metrics to be calculated are stored in a running total matrix
        met_names  = [ "Loss", "Res", "Mag", "Ang", "DLin" ]
        run_totals = np.zeros( ( 1+self.n_bins, 1+len(met_names) ) ) ## rows = (total, bin1, bin2...) x cols = (n_events, *met_names)

        ## The also build a histogram of the output magnitude
        hist = np.zeros( self.n_bins )

        ## Iterate through the validation set
        with T.no_grad():
            self.valid_loader.dataset.weight_off() ## Turn off weighted sampling!
            self.network.eval()
            for (inputs, targets, weights) in tqdm( self.valid_loader, desc="Performance", ncols=80 ):

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

                ## Get the bin numbers from the true met magnitude (final bin includes overflow)
                bins = T.clamp( targ_mag * self.n_bins / self.bin_max, 0, self.n_bins-1 ).int().cpu().numpy()

                ## Calculate the batch totals of each metric
                batch_totals = T.zeros( ( len(inputs), 1+len(met_names) ) )
                dot = T.sum( real_outp * real_targ, dim=1 ) / ( outp_mag * targ_mag + 1e-4 )

                batch_totals[:, 0] = T.ones_like(targ_mag)                                             ## Ones are for counting bins
                batch_totals[:, 1] = F.smooth_l1_loss(outputs, targets, reduction="none").mean(dim=1)  ## Loss
                batch_totals[:, 2] = ( ( real_outp - real_targ )**2 ).mean(dim=1)                      ## XY Resolution
                batch_totals[:, 3] = ( outp_mag - targ_mag )**2                                        ## Magnitude Resolution
                batch_totals[:, 4] = T.acos( dot )**2                                                  ## Angular resolution
                batch_totals[:, 5] = ( outp_mag - targ_mag ) / ( targ_mag + 1e-4 )                     ## Deviation from Linearity

                ## Fill in running data by summing over each bin, bin 0 is reserved for dataset totals
                for b in range(self.n_bins):
                    run_totals[b+1] += batch_totals[ bins==b ].sum(axis=0).cpu().numpy()

                ## Fill in the reconstructed magnitude histogram
                hist += np.histogram( outp_mag.cpu(), bins=self.n_bins, range=[0,self.bin_max] )[0]

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
        hist = np.expand_dims(hist, 0)
        hcols = [ "hist"+str(i) for i in range(self.n_bins) ]
        hdf = pd.DataFrame( data=hist, index=[self.name], columns=hcols )

        ## Write the combined dataframe to the csv
        df = pd.concat( [ mdf, hdf, self.get_info() ], axis=1 ) ## Combine the performance dataframe with info on the network
        fnm = Path( self.save_dir, self.name, "perf.csv" )
        df.to_csv( fnm, mode="w" )

        ## Turn the sampler back on for the validation set
        self.valid_loader.dataset.weight_on()

    def get_info(self):
        """ This function is used to return preselected information about the network and
            the training using the class attributes. It is use primarily for the parallel coordinate plot.
            It returns a dataframe.
        """
        columns = [ "do_rot",
                    "weight_to",
                    "weight_ratio",
                    "weight_shift",
                    "v_frac",
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
                    "epochs_trained",
                    "best_epoch" ]
        data = [[ self.__dict__[c] for c in columns ]]
        return pd.DataFrame(data=data, index=[self.name], columns=columns)
