import time
import h5py
import random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from itertools import count

import torch as T
from torch.utils.data import Dataset, IterableDataset,

def chunk_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

def buildTrainAndValidation( data_dir, test_frac ):

    ## Search the directory for HDF files
    file_list = [f for f in data_dir.glob("*.h5")]

    ## Exit if no files can be found
    if len(file_list) == 0:
        print("No files could be found with the search tag: ", data_dir, "*.h5" )
        exit()

    ## Shuffle with the a set random seed
    np.random.seed(0)
    np.random.shuffle(file_list)

    ## Split the file list according to the test_frac
    n_test  = np.clip( int(round(len(file_list)*test_frac)), 1, len(file_list)-1 )
    n_train = len(file_list) - n_test
    train_files = file_list[:-n_test]
    test_files  = file_list[-n_test:]

    return train_files, test_files

class StreamMETDataset(IterableDataset):
    def __init__(self, file_list, n_ofiles, chnk_size ):
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

        """

        ## Make attributes from all arguments
        self.file_list    = file_list
        self.n_ofiles     = n_ofiles
        self.chnk_size    = chnk_size

        ## Calculate the number of samples in the entire dataset (this step is fairly quick)
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

            This function is automatically called SEPARATELY for each thread!
            Think of it as a worker initialise function.
        """

        ## Get the current worker (thread) info
        worker_info = T.utils.data.get_worker_info()

        ## If it is None, we are doing single process loading, worker uses whole file list
        if worker_info is None:
            worker_files = self.file_list

        ## For multiple workers (threads) we break up the file list so they each work on a single subset
        else:
            worker_files = np.array_split( self.file_list, worker_info.num_workers )[worker_info.id]

        ## Further partition the worker's file list into the ones open at a time
        ofiles_list = chunk_given_size( worker_files, self.n_ofiles )

        ## We iterate through the open files collection
        for ofiles in ofiles_list:

            ## We iterate through the chunks taken from the files
            for c_count in count():

                ## Fill the buffer with the next set of chunks from the files
                buffer = self.load_chunks( ofiles, c_count )

                ## If the returned buffer is None it means that no more data could be found in the ofiles
                if buffer is None:
                    break

                ## Iterate through the samples taken from the buffer
                for sample in buffer:

                    ## Get the input, target from the row information
                    inputs  = sample[:-3]
                    targets = sample[:-1]

                    yield inputs, targets

    def load_chunks(self, files, c_count):

        ## Work out the bounds of the new chunk from within the file
        start = c_count * self.chnk_size
        end   = start + self.chnk_size

        ## Get a chunk from each file to load into the buffer
        buffer = []
        for f in files:

            ## Running "with" ensures the file is closed
            with h5py.File( f, 'r' ) as hf:

                ## Will be a 2x2 numpy array, empty if we asked for idx outside filesize
                chunk = hf["data/table"][start:end]["values_block_0"]

            ## If the chunk is not empty we add it to the buffer
            if len(chunk) != 0:
                buffer.append(chunk)

        ## If the buffer is empty it means that no files had any data left
        ## "not list" is a quicker python way to check if empty
        if not buffer:
            return None

        ## Get the buffer as a flattend (3D->2D) list
        ## Using lists like this is actually faster than numpy arrays (tested)
        buffer = [ sample for chunk in buffer for sample in chunk ]

        ## Shuffle the buffer inplace
        np.random.shuffle( buffer )

        return buffer

    def shuffle_files(self):
        """ This function shuffles the file list and must be called at the end of each epoch
            This ensures that each worker is allocated a different set of files each time and
            pseudo-shuffles the dataset over the course of training.
        """
        np.random.shuffle( self.file_list )
