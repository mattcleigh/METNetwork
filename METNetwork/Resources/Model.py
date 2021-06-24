import numpy as np
import pandas as pd
import geomloss as gl
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from itertools import count

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import METNetwork.Resources.Utils as myUT
import METNetwork.Resources.Plotting as myPL
import METNetwork.Resources.Networks as myNN
import METNetwork.Resources.Datasets as myDS

class METNET_Agent:
    def __init__(self, name, save_dir):
        print('Initialising the model')

        ## The name and save directory for outputing the model
        self.name = name
        self.save_dir = save_dir

        ## Bins for performance metrics (keep fixed for now)
        self.n_bins = 50
        self.bin_max = 400
        self.bin_max2D = 200

    def setup_network(self, input_list, act, depth, width, skips, nrm, drpt, dev = 'auto' ):
        """ This initialises the mlp network structure
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        ## Creating the neural network
        self.network = myNN.MET_MLP( "res_mlp", cut_calo, cut_track, n_in, act, depth, width, skips, nrm, drpt, dev )

    def setup_dataset( self, data_dir = '',
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

        ## Get the iterable dataset objects
        train_set = myDS.StreamMETDataset( train_files, n_ofiles, chnk_size, hist_file, weight_to, weight_ratio, weight_shift )
        valid_set = myDS.StreamMETDataset( valid_files, n_ofiles, chnk_size, hist_file, weight_to, weight_ratio, weight_shift )
        self.n_train_files = len(train_files)
        self.n_valid_files = len(valid_files)

        ## Create the pytorch dataloaders (works for both types of datset)
        self.train_loader = DataLoader( train_set,
                                        drop_last   = True,                                ## Causes errors with batch norm if batch = 1
                                        batch_size  = batch_size,                          ## Set by training script
                                        num_workers = min(self.n_train_files, n_workers),  ## Dont want dead workers if not enough files
                                        pin_memory  = True )                               ## Improves CPU->GPU transfer times
        self.valid_loader = DataLoader( valid_set,
                                        drop_last   = True,
                                        batch_size  = batch_size,
                                        num_workers = min(self.n_valid_files, n_workers),
                                        pin_memory  = True )

        ## Report on the number of files/samples used
        self.train_size = len(train_set)
        self.valid_size = len(valid_set)
        print( "Train set:       {:10} samples in {:4} files".format( self.train_size, self.n_train_files ) )
        print( "Validation set:  {:10} samples in {:4} files".format( self.valid_size, self.n_valid_files ) )

    def setup_training( self, loss_fn, lr, grad_clip, skn_weight ):
        """ Sets up variables used for training, including the optimiser
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        ## Initialise the optimiser
        self.optimiser = optim.Adam( self.network.parameters(), lr=lr )

        ## The sinkhorn loss
        self.do_skn = ( skn_weight > 0 )
        self.dist_loss = gl.SamplesLoss( "sinkhorn" )

        ## The history of train and validation losses
        self.trn_hist = []
        self.vld_hist = []
        self.epochs_trained = 0

        ## The graphs for the loss terms
        self.tot_loss_plot = myPL.loss_plot( title = self.name, xlbl = "Epoch", ylbl = "total loss" )
        self.rec_loss_plot = myPL.loss_plot( title = self.name, xlbl = "Epoch", ylbl = "reconstruction loss" )
        self.skn_loss_plot = myPL.loss_plot( title = self.name, xlbl = "Epoch", ylbl = "sinkhorn loss" )

    def _epoch( self, is_train = False ):
        """ This function performs one epoch of training on data provided by the train_loader
            It will also update the graphs after a certain number of batches pass
        """
        ## Put the nework into training mode (for batch_norm/droput)
        if is_train:
            flag = "Training"
            loader = self.train_loader
            T.enable_grad()
            self.network.train()
        else:
            flag = "Testing"
            loader = self.valid_loader
            T.no_grad()
            self.network.eval()

        running_loss = np.zeros(3)
        for i, (inputs, targets, weights) in enumerate( tqdm( loader, desc=flag, ncols=80 ) ):

            ## Zero out the gradients
            if is_train:
                self.optimiser.zero_grad()

            ## Move data to the network device (GPU)
            inputs = inputs.to(self.network.device)
            targets = targets.to(self.network.device)
            weights = weights.to(self.network.device)

            ## Calculate the network output
            outputs = self.network( inputs )

            ## Calculate the weighted batch reconstruction loss
            rec_loss = ( self.loss_fn( outputs, targets ).mean( dim=1 ) * weights ).mean()

            ## Calculate the sinkhorn loss if required
            skn_loss = T.zeros_like(rec_loss)
            if self.do_skn:
                skn_loss = self.dist_loss( outputs, targets )

            ## Combine the losses
            tot_loss = rec_loss + self.skn_weight * skn_loss

            ## Calculate the gradients and update the parameters
            if is_train:
                tot_loss.backward()
                if self.grad_clip > 0: nn.utils.clip_grad_norm_( self.network.parameters(), self.grad_clip )
                self.optimiser.step()

            ## Update the running loss
            new_loss = np.array( [ tot_loss.item(), rec_loss.item(), skn_loss.item() ] )
            running_loss = myUT.update_avg( running_loss, new_loss, i+1 )

        ## Calculate the mean of the losses (by i) and update
        if is_train:
            self.trn_hist.append( list( running_loss ) )
            self.epochs_trained += 1
        else:
            self.vld_hist.append( list( running_loss ) )

    def run_training_loop( self, patience=20 ):
        """ This is the main loop which cycles epochs of train and test
            It saves the network and checks for early stopping
        """
        ## Update our information dictionary
        self.__dict__.update( (k,v) for k,v in vars().items() if k != "self" )

        for epc in count(self., max_epochs+1):

            print( "\nEpoch: {}".format(epoch) )

            ## Run the test/train cycle
            self._epoch( is_train = True )
            self._epoch( is_train = False )

            ## Shuffle the file order in the datasets
            self.train_loader.dataset.shuffle_files()
            self.valid_loader.dataset.shuffle_files()

            ## Calculate the best epoch and the number of bad epochs
            self.best_epoch = np.argmin( np.array(self.vld_hist)[:, 0] ) + 1
            self.bad_epochs = len( self.vld_hist) - self.best_epoch

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

        ## Save the loss history
        hist_array = np.concatenate( [self.trn_hist, self.vld_hist], axis=-1 )
        np.savetxt( Path( full_name, "train_hist.csv" ), hist_array )

        ## Calculate the losses as numpy arrays for slicing
        trn = np.array( self.trn_hist )
        vld = np.array( self.vld_hist )

        ## Save the loss plots
        self.tot_loss_plot.save( trn[:, 0], vld[:, 0], Path( full_name, "loss_tot.png" ) )
        self.rec_loss_plot.save( trn[:, 1], vld[:, 1], Path( full_name, "loss_rec.png" ) )
        self.skn_loss_plot.save( trn[:, 2], vld[:, 2], Path( full_name, "loss_skn.png" ) )

        ## Save a copy of the stat file in the network directory: Only on the first iteration!
        if self.epochs_trained==1:
            shutil.copyfile( self.stat_file, Path( full_name, "stat.csv") )

        ## If the network is the best, then we also save the performance file
        if self.bad_epochs==0:
            print("\nImprovement detected! Saving additional information")
            self.save_perf()

    def load( self, flag, get_opt = True ):

        ## Get the name of the directories
        full_name = Path( self.save_dir, self.name )
        model_folder = Path( full_name, "models" )

        ## Load the network and optimiser
        self.network.load_state_dict( T.load( Path( model_folder, "net_"+flag ) ) )

        if get_opt:
            self.optimiser.load_state_dict( T.load( Path( model_folder, "opt_"+flag ) ) )

        ## Load the train history
        previous_data = np.loadtxt( Path( full_name, "train_hist.csv" ) )
        self.trn_hist = previous_data[:,:3].tolist()
        self.vld_hist = previous_data[:,3:].tolist()
        self.best_epoch = np.argmin( np.array(self.vld_hist)[:, 0] ) + 1
        self.epochs_trained = len(self.trn_hist)

    def save_perf( self ):

        ## Calculating performance will require un-normalising our outputs, so we need the stats
        stats = np.loadtxt(self.stat_file, skiprows=1, delimiter=",")
        means = T.from_numpy(stats[0,-2:]).to(self.network.device) ## Just need the target (True EX, EY) stats
        devs  = T.from_numpy(stats[1,-2:]).to(self.network.device)

        ## The performance metrics to be calculated are stored in a running total matrix
        met_names  = [ "Loss", "Res", "Mag", "Ang", "DLin" ]
        run_totals = np.zeros( ( 1+self.n_bins, 1+len(met_names) ) ) ## rows = (total, bin1, bin2...) x cols = (n_events, *met_names)

        ## Build a 1D and 2D histogram of the output magnitude
        hist1D = np.zeros( self.n_bins )
        hist2D = np.zeros( (self.n_bins, self.n_bins) )
        outp2D = np.zeros( (self.n_bins, self.n_bins) )

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
        hcols = [ "hist"+str(i) for i in range(self.n_bins) ]
        hdf = pd.DataFrame( data=hist1D, index=[self.name], columns=hcols )

        ## Write the combined dataframe to the csv
        df = pd.concat( [ mdf, hdf, self.get_info() ], axis=1 ) ## Combine the performance dataframe with info on the network
        fnm = Path( self.save_dir, self.name, "perf.csv" )
        df.to_csv( fnm, mode="w" )

        ## Save the 2D histogram to the output folder
        hnm = Path( self.save_dir, self.name, "hist2d.png" )
        myPL.save_hist2D( hist2D, hnm, [0, self.bin_max2D, 0, self.bin_max2D ], [[0,self.bin_max2D],[0,self.bin_max2D]] )

        hnm = Path( self.save_dir, self.name, "outp2d.png" )
        myPL.save_hist2D( outp2D, hnm, [-100, 500, -200, 200 ], [[-100,500],[0,0]] )

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
                    "skn_weight",
                    "epochs_trained",
                    "best_epoch",
                    "cut_calo",
                    "cut_track", ]
        data = [[ self.__dict__[c] for c in columns ]]
        return pd.DataFrame(data=data, index=[self.name], columns=columns)
