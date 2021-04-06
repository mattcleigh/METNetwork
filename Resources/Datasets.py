import h5py
import random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from itertools import count

import torch as T
from torch.utils.data import IterableDataset

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

class METDataset(IterableDataset):
    def __init__(self, file_list, n_ofiles, chnk_size, weight_to):

        ## Make attributes from all arguments
        self.file_list  = file_list
        self.n_ofiles   = n_ofiles
        self.chnk_size  = chnk_size
        self.weight_to  = weight_to
        self.do_weights = True if weight_to > 0 else False

        ## Calculate the number of samples in the dataset set
        self.n_samples = 0
        for file in tqdm( self.file_list, desc="Collecting Files", ncols=80, unit="" ):
            hf = h5py.File( file, 'r' )
            file_len = len(hf["data/table"])
            hf.close()
            self.n_samples += file_len

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
            wrk_id = worker_info.id
            n_wrks = worker_info.num_workers
            worker_files = np.array_split( self.file_list, n_wrks )[wrk_id]

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
                for sample in buffer:

                    ## Get the MLP input and target variables
                    inputs  = sample[:-3]
                    targets = sample[-3:-1]
                    weight  = sample[-1]

                    ## We return the sample if the weights allow it (or statements are short circuted)
                    if not self.do_weights or weight > self.weight_to or self.weight_to*random.random() < weight:
                        yield inputs, targets

    def load_chunks(self, files, c_count):

        ## Work out the bounds of the new chunk
        start = c_count * self.chnk_size
        end   = start + self.chnk_size

        ## Get a chunk from each file
        buffer = []
        for f in files:

            ## Running "with" ensures the file is closed
            with h5py.File( f, 'r' ) as hf:
                chunk = hf["data/table"][start:end]     ## Will return an empty list if we asked for idx outside filesize
                chunk = np.array([c[1] for c in chunk]) ## Annoying thing we must do as the data is a tuple (idx, [sample])

            ## If the chunk is non empty we add it to the buffer
            if len(chunk) != 0:
                buffer.append(chunk)

        ## If the buffer is empty it means that no files had any data left
        if len(buffer) == 0:
            return None

        ## Get the buffer as a single array and shuffle
        buffer = np.vstack( buffer )
        np.random.shuffle( buffer )
        return buffer

    def shuffle_files(self):
        ## We shuffle the file list, this is called at the end of each epoch
        np.random.shuffle( self.file_list )

    def weight_on(self):
        self.do_weights = True

    def weight_off(self):
        self.do_weights = False
