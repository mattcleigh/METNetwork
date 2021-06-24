import time
import h5py
import random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from itertools import count
from scipy.interpolate import interp1d

import torch as T
from torch.utils.data import Dataset, IterableDataset

from METNetwork.Resources import Utils as myUT

def buildTrainAndValidation( data_dir, test_frac ):

    ## Search the directory for HDF files
    file_list = [f for f in data_dir.glob("*.h5")]

    ## Exit if no files can be found
    if len(file_list) == 0:
        raise LookupError("No files could be found with the search tag: ", data_dir, "*.h5" )

    ## Shuffle with the a set random seed
    np.random.seed(0)
    np.random.shuffle(file_list)

    ## Split the file list according to the test_frac
    n_test  = np.clip( int(round(len(file_list)*test_frac)), 1, len(file_list)-1 )
    train_files = file_list[:-n_test]
    test_files  = file_list[-n_test:]

    return train_files, test_files

class StreamMETDataset(IterableDataset):
    def __init__(self, file_list, n_ofiles, chnk_size, hist_file, weight_to, weight_ratio, weight_shift ):
        """ An iterable dataset for when the training set is too large to hold in memory.

            Works with multithreading:
             - Each thread has a assigned multiple hdf files that it will work on over the epoch (its worker_files).
                - Each thread selects a set number of files to have open at a time (its ofiles_list).
                    - Each thread reads a chunk from each of its open files to fill a buffer which is shuffled.
                        - Each thread then iterates through the buffer
                    - When the buffer is empty the thread loads new chucks from its o_files
                - When the o_files are empty then it opens a new set from its file list
             - When the file list is empty then the thread is finished for its epoch

            Minimal memory footprint. Amount of data stored in memory at given time is:
                - sample_size x chunk_size x n_ofiles x n_threads
                                 ^  ( buffer_size ) ^
        Args:
            file_list: A python list of file names (with directories) to open for the epoch
            n_ofiles:  An int of the number of files to read from simultaneously
                       Larger n_ofiles means that the suffling between epochs is closer to a real shuffling
                       of the dataset, but it will result in more memory used.
            chnk_size: The size of the chunk to read from each of the ofiles.
            other:     Arguments solely for the Weight_Function class

        """

        ## Class attributes
        self.file_list    = file_list
        self.n_ofiles     = n_ofiles
        self.chnk_size    = chnk_size
        self.weight_exist = ( weight_to + weight_shift ) != 0 ## This is fixed
        self.do_weights   = self.weight_exist ## This is toggled on and off for validation

        ## We load the function that calculates weight based on True Et
        self.WF = myUT.Weight_Function( hist_file, weight_to, weight_ratio, weight_shift )

        ## Calculate the number of samples in the dataset set
        self.n_samples = 0
        for file in tqdm( self.file_list, desc="Collecting Files", ncols=80, unit="" ):
            with h5py.File( file, 'r' ) as hf:
                self.n_samples += len(hf["data/table"])

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        """ This function is called whenever an iterator is created on the
            dataloader responsible for this training set.
            ie: Every "for batch in dataloader" call

            This function is called SEPARATELY for each thread
            Think of it as a worker initialise function
        """

        ## Get the worker info
        worker_info = T.utils.data.get_worker_info()

        ## If it is None, we are doing single process loading, worker uses whole file list
        if worker_info is None:
            worker_files = self.file_list

        ## For multiple workers we break up the file list so they each work on a single subset
        else:
            worker_files = np.array_split( self.file_list, worker_info.num_workers )[worker_info.id]

        ## Further partition the worker's file list into the ones open at a time
        ofiles_list = myUT.chunk_given_size( worker_files, self.n_ofiles )

        ## We iterate through the open files collection
        for ofiles in ofiles_list:

            ## We iterate through the chunks taken from the files
            for c_count in count():

                ## Fill the buffer with the next set of chunks from the files
                buffer = self.load_chunks( ofiles, c_count )

                ## If the returned buffer is None it means that no more data could be found in the ofiles
                if buffer is None:
                    break

                ## Iterate through the batches taken from the buffer
                for sample in buffer: ## sample is [inputs, targT, targx, targy, weight]

                    ## Get the input, target, and sample weight
                    inputs  = sample[:-4]
                    targets = sample[-3:-1]
                    weight  = sample[-1]

                    ## Always return with weight=1 if no weighting is applied
                    if not self.do_weights:
                        yield inputs, targets, 1

                    ## We check if we want to return the weight for the loss function
                    elif self.WF.thresh <= weight:
                            yield inputs, targets, weight

                    ## Otherwise we downsample
                    elif self.WF.thresh*random.random() <= weight:
                            yield inputs, targets, self.WF.thresh

    def load_chunks(self, files, c_count):

        ## Work out the bounds of the new chunk
        start = c_count * self.chnk_size
        end   = start + self.chnk_size

        ## Get a chunk from each file to load into the buffer
        buffer = []
        for f in files:

            ## Running "with" ensures the file is closed
            with h5py.File( f, 'r' ) as hf:

                ## A 2x2 numpy array, empty if we asked for idx outside filesize
                chunk = hf["data/table"][start:end]["values_block_0"]

            ## If the chunk is not empty we add it to the buffer
            if len(chunk):

                ## Add a column of weights to the chunk based on True_ET magnitude
                ws = self.WF.apply( chunk[:, -3:-2] ) if self.do_weights else np.ones( (len(chunk),1) )
                chunk = np.concatenate( [chunk, ws.astype("float32") ], axis=-1 )
                buffer.append(chunk)

        ## If the buffer is empty it means that no files had any data left ("not list" is a quicker python way to check if empty)
        if not buffer:
            return None

        ## Get the buffer as a flattend (3D->2D) list and shuffle (using lists is faster than numpy arrays TESTED)
        buffer = [ sample for chunk in buffer for sample in chunk ]
        np.random.shuffle( buffer )

        return buffer

    def shuffle_files(self):
        ## We shuffle the file list, this is called at the end of each epoch
        np.random.shuffle( self.file_list )

    def weight_on(self):
        if self.weight_exist:
            self.do_weights = True

    def weight_off(self):
        self.do_weights = False
