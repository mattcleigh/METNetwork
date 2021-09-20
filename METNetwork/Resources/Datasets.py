import time
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import count

import torch as T
from torch.utils.data import IterableDataset

import METNetwork.Resources.Utils as myUT
import METNetwork.Resources.Weights as myWT

def buildTrainAndValid(data_dir, v_frac):
    """
    Returns a list of files in data_dir to be used for training and validation.
    This is required for pytorch iterable datasets which can not be ramdomly divided, as each process thread
    needs to run over an entire HDF file. Therefore the splitting is done per file rather than per sample.
    The files however are shuffled beforehand and there are ususally 1k HDF files that make up a dataset, so this
    splitting should be reasonable.
    args:
        data_dir: The directory to look for *h5 files
        v_frac: The fraction of files to be used for validation
    """
    ## Search the directory for HDF files
    file_list = [f for f in Path(data_dir).glob('*.h5')]
    n_files = len(file_list)

    ## Exit if no files can be found
    if n_files == 0:
        raise LookupError('No HDF files could be found in ', data_dir)
    ## Shuffle with the a set random seed
    np.random.seed(42)
    np.random.shuffle(file_list)

    ## Split the file list according to the vaid_frac (must have at least 1 train and valid!)
    n_valid  = np.clip( int(n_files*v_frac), 1, n_files-1 )
    valid_files = file_list[:n_valid]
    train_files = file_list[n_valid:] ## Hacky solution to allow plotting method to use all valid

    return train_files, valid_files

class StreamMETDataset(IterableDataset):
    def __init__(self, file_list, var_list, n_ofiles, chnk_size, weight_type, weight_to, weight_shift, weight_ratio):
        """
        An iterable dataset for when the training set is too large to hold in memory.
        Also applies a weight for each event, which is either used for sampling or for use in the loss function

        Works with multithreading.
        Epoch start:
         - Each thread is assigned a mututally explusive collection of HDF files (its worker_files).
         - Each thread groups its assigned files into mini collections of size n_ofiles (its ofiles_list).
         - Each thread loops through its mini collection of files
             - It reads chunk of data from each file in the mini collection and fills a buffer (shuffled).
                - It calculates sample weights for the whole buffer
                    - It loops through the buffer and weights, yeilding samples
                - When the buffer is empty the thread loads new chucks from each file in the current mini collection
             - When the mini collection is empty it moves to the next one are empty then it opens a new set from its file list
         - When the file list is empty then the thread is finished for its epoch

        Minimal memory footprint. Amount of data stored in memory at given time is:
            - sample_size x chunk_size x n_ofiles x n_threads
                             ^  ( buffer_size ) ^
        Args:
            file_list: A python list of file names (with directories) to open for the epoch
            var_list:  A list of strings indicating which variables should be loaded from each HDF file
            n_ofiles:  An int of the number of files to read from simultaneously
                       Larger n_ofiles means that the suffling between epochs is closer to a real shuffling
                       of the dataset, but it will result in more memory used.
            chnk_size: The size of the chunk to read from each of the ofiles.
            other:     Arguments solely for the SampleWeight class
        """

        ## Class attributes
        self.file_list = file_list
        self.var_list = var_list
        self.n_ofiles = n_ofiles
        self.chnk_size = chnk_size

        ## Iterate through a files and calculate the number of events
        self.n_samples = 0
        for file in tqdm(self.file_list, desc='Collecting Files', unit='', ascii=True):
            with h5py.File(file, 'r') as hf:
                self.n_samples += len(hf['data/table'])

        ## Booleans indicating whether we need to be calculating and applying event weights
        self.weight_exist = bool(weight_to) or bool(weight_shift) ## Fixed for duration of the class
        self.do_weights = self.weight_exist ## Toggled on and off for performance testing

        ## Initialise a class which calculates a per event weight based on some histogram in the training folder
        if self.weight_exist:
            folder = file_list[0].parent.absolute()
            self.SW = myWT.SampleWeight(folder, weight_type, weight_to, weight_shift, weight_ratio)

    def shuffle_files(self):
        '''
        Shuffles the entire file list, meaning that each worker gets a different subset
        Should be called inbetween each epoch call
        '''
        np.random.shuffle(self.file_list)

    def weight_on(self):
        '''
        Turns on calculating per sample weights
        Needed for the training and validation epochs
        '''
        self.do_weights = self.weight_exist

    def weight_off(self):
        '''
        Turns off calculating per sample weights
        Needed for the performance evaluation steps
        '''
        self.do_weights = False

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        '''
        Called automatically whenever an iterator is created on the
        dataloader responsible for this dataset set.
        ie: Whenever 'for batch in dataloader: ...' is executed

        This function is called SEPARATELY for each thread
        Think of it as a worker (thread) initialise function
        '''

        ## Get the worker info
        worker_info = T.utils.data.get_worker_info()

        ## If it is None we are doing single process loading, worker uses whole file list
        if worker_info is None:
            worker_files = self.file_list

        ## For multiple workers we break up the file list so they each work on a single subset
        else:
            worker_files = np.array_split(self.file_list, worker_info.num_workers)[worker_info.id]

        ## Further partition the worker's file list into the ones it open at a time
        ofiles_list = myUT.chunk_given_size(worker_files, self.n_ofiles)

        ## We iterate through the open files collection
        for ofiles in ofiles_list:

            ## We iterate through the chunks taken from each of the files
            for c_count in count():

                ## Fill the buffer with the next set of chunks from the files
                buffer = self.load_chunks(ofiles, c_count)

                ## If the returned buffer is empty there are no more events in these files!
                if not buffer.size:
                    break

                ## Calculate all weights for the buffer using the final 3 variables (truth) in one go
                weights = self.SW.apply(buffer[:, -3:]) if self.do_weights else np.ones(len(buffer))

                ## Iterate through the samples taken from the buffer
                for sample, weight in zip(buffer, weights):

                    ## Yield the event if the weight is non-zero (skips the event otherwise)
                    if weight:
                        yield sample[:-3], sample[-2:], weight

    def load_chunks(self, files, c_count):

        ## Work out the bounds of the new chunk within the file
        start = c_count * self.chnk_size
        stop = start + self.chnk_size

        buffer = []
        for f in files:
            hf = h5py.File(f, 'r') ## Open the hdf file
            chunk = hf['data/table'][start:stop][self.var_list] ## Returns np array of tuples
            buffer += [ list(event) for event in chunk ] ## Converts into a list of lists
            hf.close() ## Close the hdf file

        ## Shuffle and return the buffer as a numpy array so it works with weighting and the dataloader's collate function
        np.random.shuffle(buffer)
        return np.array(buffer, dtype=np.float32)
