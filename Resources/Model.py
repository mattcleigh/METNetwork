import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Plotting as myPL
from Resources import Networks as myNN
from Resources import Datasets as myDS

import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import count

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class METNET_Agent(object):
    def __init__(self, name, save_dir ):
        self.name = os.path.join( save_dir, name )

    def setup_dataset( self, train_files, test_files, stat_file, n_ofiles, chnk_size, batch_size, n_workers ):
        """ Uses a pytorch dataloader to get the training and testing sets
        """
        ## Load the stat file and the x indeces for pre-processing
        self.stat_file = stat_file

        ## Get the defined dataset objects (created from multiple files)
        train_set = myDS.METDataset( train_files, n_ofiles, chnk_size )
        test_set  = myDS.METDataset( test_files,  n_ofiles, chnk_size )

        ## Create the pytorch dataloaders
        self.train_loader = DataLoader( train_set, batch_size=batch_size, num_workers=n_workers )
        self.test_loader  = DataLoader( test_set,  batch_size=batch_size, num_workers=n_workers )

        ## Report on the number of files used
        print( "# Files used in train set: ", len(train_set.file_list) )
        print( "# Files used in test set:  ", len(test_set.file_list) )

    def setup_network( self, act, depth, width, skips, nrm, drpt ):
        """ This initialises the mlp network structure
        """

        ## Creating the neural network
        self.network = myNN.MET_MLP( "res_mlp", act, depth, width, skips, nrm, drpt )

    def setup_training( self, loss_fn, lr, clip_grad, interactive ):
        """ Sets up variables used for training, including the optimiser
        """

        ## Initialise attributes needed for training
        self.loss_fn = loss_fn
        self.optimiser = optim.Adam( self.network.parameters(), lr=lr )
        self.clip_grad = clip_grad
        self.ion = interactive

        self.trn_hist = []
        self.tst_hist = []
        self.tim_hist = []
        self.epochs_trained = 0

        if self.ion:
            plt.ion()
            self.loss_plot = myPL.loss_plot( self.name )

    def training_epoch( self ):
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
        self.trn_hist.append( running_loss / i )
        self.epochs_trained += 1

    def testing_epoch(self):
        """ This function performs one epoch of testing on data provided by the test_loader
        """

        with T.no_grad():
            self.network.eval()
            running_loss = 0
            for i, (inputs, targets) in enumerate( tqdm( self.test_loader, desc="Testing ", ncols=80, unit="" ) ):
                inputs = inputs.to(self.network.device)
                targets = targets.to(self.network.device)
                outputs = self.network( inputs )
                loss = self.loss_fn( outputs, targets )
                running_loss += loss.item()
            self.tst_hist.append( running_loss / i )

    def run_training_loop( self, patience = 20 ):
        """ This is the main loop which cycles epochs of train and test
            It also updates graphs and saves the network
        """

        for epoch in count(1):

            print( "\nEpoch: {}".format(epoch) )

            ## For time keeping
            e_start_time = time.time()

            ## Run the test/train cycle
            self.testing_epoch()
            self.training_epoch()

            ## Shuffle the file order in the datasets
            self.train_loader.dataset.shuffle_files()
            self.test_loader.dataset.shuffle_files()

            ## For time keeping
            self.tim_hist.append( time.time() - e_start_time )

            ## Printing loss
            print( "Average loss on training set:   {:.5}".format( self.trn_hist[-1] ) )
            print( "Average loss on testing set:    {:.5}".format( self.tst_hist[-1] ) )
            print( "Epoch Time: {:.5}".format( self.tim_hist[-1] ) )

            ## Update the loss plot
            if self.ion:
                self.loss_plot.update( self.trn_hist[-500:], self.tst_hist[-500:] )

            ## Calculate the number of bad epochs
            bad_epochs = len(self.tst_hist) - np.argmin(self.tst_hist) - 1

            ## Saving the networks
            self.save("latest")
            if bad_epochs==0:
                self.save("best")
            else:
                print( "Bad Epoch Number: {:}".format( bad_epochs ) )
                if bad_epochs > patience:
                    print( "Patience Exceeded: Stopping training!" )
                    return 0

    def save( self, flag = "" ):

        ## Create the save directory if it doesnt exits yet
        if not os.path.exists( self.name ):
            os.system( "mkdir -p {}".format(self.name) )

        ## Save a copy of the stat file in the network directory
        if self.epochs_trained==1:
            if self.stat_file is not None:
                shutil.copyfile( self.stat_file, os.path.join( self.name, "stats.csv" ) )

        ## Save the network and optimiser
        T.save( self.network.state_dict(), os.path.join( self.name, self.network.name + "_" + flag ) )
        T.save( self.optimiser.state_dict(), os.path.join( self.name, "optimiser_" + flag ) )

        ## Saving the training history
        if flag=="latest":
            out_array = np.transpose( np.vstack(( self.trn_hist, self.tst_hist, self.tim_hist )) )
            np.savetxt( os.path.join( self.name, "train_hist.csv" ), out_array )

    def load( self, flag = "" ):

        ## Load the network and optimiser
        self.network.load_state_dict( T.load( os.path.join( self.name, self.network.name + "_" + flag ) ) )
        self.optimiser.load_state_dict( T.load( os.path.join( self.name, "optimiser_" + flag ) ) )

        ## Load the train history
        previous_data = np.loadtxt( os.path.join( self.name, "train_hist.csv" ), header=None )
