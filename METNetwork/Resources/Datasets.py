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
    np.random.seed(0)
    np.random.shuffle(file_list)

    ## Split the file list according to the vaid_frac (must have at least 1 train and valid!)
    n_valid  = np.clip(int(n_files*v_frac), 1, n_files-1)
    valid_files = file_list[:n_valid]
    train_files = file_list[n_valid:]

    return train_files, valid_files

class StreamMETDataset(IterableDataset):
    def __init__(self, file_list, inpt_list, n_ofiles, chnk_size, weight_to, weight_ratio, weight_shift):
        """
        An iterable dataset for when the training set is too large to hold in memory.
        Also applies a weight for each event, which is either used for sampling or for use in the loss function

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
            inpt_list: A list of strings indicating which input variables should be loaded from memory
            n_ofiles:  An int of the number of files to read from simultaneously
                       Larger n_ofiles means that the suffling between epochs is closer to a real shuffling
                       of the dataset, but it will result in more memory used.
            chnk_size: The size of the chunk to read from each of the ofiles.
            other:     Arguments solely for the SampleWeight class
        """

        ## Class attributes
        self.file_list = file_list
        self.n_ofiles = n_ofiles
        self.chnk_size = chnk_size

        ## The list of variables to read from the hdf files, combining the input list with the target information
        self.var_list = inpt_list + ['True_EX', 'True_EY', 'Raw_True_ET', 'Raw_True_EX', 'Raw_True_EY']

        ## Booleans indicating whether we need to be calculating and applying event weights
        self.weights_exit = bool(weight_to) or bool(weight_shift) ## Fixed for duration of the class
        self.do_weights = True ## Toggled on and off for performance testing

        ## Iterate through a files and calculate the number of events
        self.n_samples = 0
        for file in tqdm(self.file_list, desc='Collecting Files', ncols=100, unit='', ascii=True):
            with h5py.File(file, 'r') as hf:
                self.n_samples += len(hf['data/table'])

        ## Initialise a class which calculates a per event weight based on a True ET miss histogram file (in same folder as data)
        if self.weights_exit:
            hist_file = Path(file_list[0].parent.absolute(), 'hist.csv')
            self.SW = myWT.SampleWeight(hist_file, weight_to, weight_ratio, weight_shift)

    def shuffle_files(self):
        np.random.shuffle(self.file_list) ## Should be called before iterating

    def weight_on(self):
        self.do_weights = self.weight_exist

    def weight_off(self):
        self.do_weights = False

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        """ This function is called whenever an iterator is created on the
            dataloader responsible for this training set.
            ie: Every 'for batch in dataloader' call

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
            worker_files = np.array_split(self.file_list, worker_info.num_workers)[worker_info.id]

        ## Further partition the worker's file list into the ones it open at a time
        ofiles_list = myUT.chunk_given_size(worker_files, self.n_ofiles)

        ## We iterate through the open files collection
        for ofiles in ofiles_list:

            ## We iterate through the chunks taken from each of the files
            for c_count in count():

                ## Fill the buffer with the next set of chunks from the files
                buffer = self.load_chunks(ofiles, c_count)

                ## Iterate through the batches taken from the buffer
                for sample in buffer:

                    ## Read the information contained in the sample
                    inputs  = sample[:-5]
                    targets = sample[-5:-3]
                    true_et = sample[-3:]

                    ## Perform the weighted sampling
                    if self.do_weights:                   ## Are we applying weights to our samples
                        weight = self.SW.apply(true_et)   ## Calculate the weight (0 if it failed the random threshold)
                        if weight:                        ## Yield the sample with the calculated weight
                            yield inputs, targets, weight ## Note that if this fails, then no sample is yeilded (skipped)
                    else:                                 ## Otherwise if we are not doing weights
                        yield inputs, targets, 1          ## Then always return the sample with weight = 1

    def load_chunks(self, files, c_count):

        ## Work out the bounds of the new chunk
        start = c_count * self.chnk_size
        stop = start + self.chnk_size

        buffer = []
        for f in files:
            hf = h5py.File(f, 'r') ## Open the hdf file
            chunk = hf['data/table'][start:stop][self.var_list].tolist() ## Returns a list of tuples
            buffer += [ list(event) for event in chunk ] ## Converts the tuples into a list
            hf.close() ## Close the hdf file

        ## Shuffle and return the buffer as a torch tensor so it works with the dataloader's collate function
        np.random.shuffle(buffer)
        return T.tensor(buffer)
