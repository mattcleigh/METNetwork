import time
import h5py
import random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from itertools import count
from scipy.interpolate import interp1d

import torch as T
from torch.utils.data import Dataset, IterableDataset, WeightedRandomSampler, RandomSampler, Sampler, SequentialSampler

from Resources import Utils as myUT

class BiasFunction(object):
    def __init__(self, bias):
        self.k = (1-bias)**3
        self.ret_id = True if bias == 0 else False
        self.ret_0  = True if bias == 1 else False
    def apply(self, x):
        if self.ret_id: return x
        if self.ret_0: return 0*x
        return (x * self.k) / (x * self.k - x + 1 ) ## Convex shape


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

class SwitchableSampler(Sampler):
    def __init__(self, dataset, sample_weights, n_samples, max_weight ):
        """ A dataset sampler that can switch between random sampling and weighted random sampling
            Switching is done by using the weight_on and weight_off methods.
            This sampler is used in the METDataset (map version)
        """
        self.n_samples = n_samples
        self.max_weight = max_weight
        self.do_weights = True if max_weight > 0 else False

        self.rs = RandomSampler( dataset ) ## The weighted sampler is None if we have no weights to give
        self.ws = WeightedRandomSampler( sample_weights, n_samples ) if max_weight > 0 else None

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        if self.do_weights: return iter(self.ws)
        else:               return iter(self.rs)

    def weight_on(self):
        if self.max_weight > 0:
            self.do_weights = True

    def weight_off(self):
        self.do_weights = False

class METDataset(Dataset):
    def __init__(self, file_list, max_weight):
        """ A standard mappable pytorch dataset where the entire dataset is loaded into memory.
            This results in a massive memory footprint and only should be used for testing and special cases.
            Unlike the iterable dataset, this contains a sampler attribute, which is passed to the dataloader.
            The sampler depends if we are using weights, in which case it is Switchable, otherwise it is none,
            which defaults to a random sampler.
        """

        ## Load all the samples into memory, this, size, and sampler are the only class attributes
        ## All other information is held by the sampler
        self.all_samples = []
        sample_weights = []
        for file in tqdm( file_list, desc="Loading data into memory", ncols=80, unit="" ):

            with h5py.File( file, 'r' ) as hf:
                file_data = hf["data/table"]["values_block_0"]

                ## We use list addition due to the sizes of the list at hand, append would require us flattening the data (copy)
                self.all_samples += list(file_data[:, :-1])

                ## We dont bother loading weights into memory if they will never be used
                if max_weight > 0: sample_weights += list(file_data[:, -1])

        ## Clip the weighted array using the max weight and count the number of samples
        if max_weight > 0: sample_weights = np.clip( sample_weights, 0, max_weight )
        self.n_samples = len(self.all_samples)

        ## The sampler passed to the dataloader depends on if we are using weights
        self.sampler = SwitchableSampler( self, sample_weights, self.n_samples, max_weight )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ## Get the sample from the list
        sample = self.all_samples[idx]

        ## Break into inputs, targets and weight value (sampler handles weights)
        inputs  = sample[:-2]
        targets = sample[-2:]

        return inputs, targets

    def weight_on(self):
        self.sampler.weight_on()

    def weight_off(self):
        self.sampler.weight_off()

    def shuffle_files(self): ## This is to keep the same interface as the iterable dataset
        return None

class StreamMETDataset(IterableDataset):
    def __init__(self, file_list, n_ofiles, chnk_size, hist_file, weight_type):
        """ An iterable dataset for when the trainin set is too large to hold in memory.
            It defines a buffer for each thread which reads in chunks from a set number of files.
            Minimal memory footprint. Unlike the mapable dataset, this has no sampler attribute.
            Instead the retrieval of samples is built directly in the iter method.
        """

        ## Make attributes from all arguments
        self.file_list   = file_list
        self.n_ofiles    = n_ofiles
        self.chnk_size   = chnk_size
        self.weight_type = weight_type
        self.do_weights = True if weight_type != "" else False ## This is toggled on and off for validation

        ## We load the function that calculates weight based on True Et
        self.WF = myUT.Weight_Function( weight_type, hist_file )

        ## Calculate the number of samples in the dataset set
        self.n_samples = 0
        for file in tqdm( self.file_list, desc="Collecting Files", ncols=80, unit="" ):
            with h5py.File( file, 'r' ) as hf:
                self.n_samples += len(hf["data/table"])

        ## Setting the unwanted attributes to dataloader defaults so that we have same interface as METDataset
        self.sampler = None
        self.shuffle = False

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

                ## Iterate through the batches taken from the buffer
                for sample in buffer: ## sample is [inputs, targx, targy, weight]

                    ## Get the input, target, and sample weight
                    inputs  = sample[:-3]
                    targets = sample[-3:-1]
                    weight  = sample[-1]

                    ## We return with weight one if no weighting is applied
                    if not self.do_weights:
                        yield inputs, targets, 1

                    ## We check if we want to return the weight
                    elif self.WF.sweight < weight:
                        yield inputs, targets, weight

                    ## Otherwise we downsample
                    elif self.WF.sweight*random.random() <= weight:
                        yield inputs, targets, self.WF.sweight



    def load_chunks(self, files, c_count):

        ## Work out the bounds of the new chunk
        start = c_count * self.chnk_size
        end   = start + self.chnk_size

        ## Get a chunk from each file to load into the buffer
        buffer = []
        for f in files:

            ## Running "with" ensures the file is closed
            with h5py.File( f, 'r' ) as hf:

                ## Will a 2x2 numpy array, empty if we asked for idx outside filesize
                chunk = hf["data/table"][start:end]["values_block_0"]

            ## If the chunk is not empty we add it to the buffer
            if len(chunk) != 0:

                ## Replace the last column from True_Et miss values to weights based on those values
                if self.do_weights:
                    chunk[:, -1] = self.WF.apply( chunk[:, -1] )
                buffer.append(chunk)

        ## If the buffer is empty it means that no files had any data left ("not list" is a quicker python way to check if empty)
        if not buffer:
            return None

        ## Get the buffer as a flattend (3D->2D) list and shuffle (using lists is faster than numpy arrays)
        buffer = [ sample for chunk in buffer for sample in chunk ]
        np.random.shuffle( buffer )

        return buffer

    def shuffle_files(self):
        ## We shuffle the file list, this is called at the end of each epoch
        np.random.shuffle( self.file_list )

    def weight_on(self):
        if self.weight_type != "":
            self.do_weights = True

    def weight_off(self):
        self.do_weights = False
