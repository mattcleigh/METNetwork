import sys
home_env = '../'
sys.path.append(home_env)

import glob
import h5py
import numpy as np
import numpy.random as rd
from itertools import count

import torch as T
import torchvision as TV
from torch.utils.data import IterableDataset

class METDataset(IterableDataset):
    def __init__(self, file_names, n_ofiles, chnk_size):
        self.file_list = np.array(glob.glob(file_names))
        self.n_ofiles  = n_ofiles
        self.chnk_size = chnk_size

    def __iter__(self):
        """ This function is called whenever an iterator is created on the
            dataloader responsible for this training set.
            ie: Every "for batch in dataloader" call

            This function is called SEPARATELY for each thread
            Think of it as a worker initialise function
        """

        ## Get the worker info
        worker_info = T.utils.data.get_worker_info()

        ## If it is None, we are doing single process loading
        if worker_info is None:
            worker_files = self.file_list

        else:
            wrk_id = worker_info.id
            n_wrks = worker_info.num_workers
            worker_files = np.array_split( self.file_list, n_wrks )[wrk_id]

        ## Further split the file list into the ones open at a time
        n_splits = len(worker_files) / self.n_ofiles
        ofiles_list = np.array_split(worker_files, n_splits)

        ## We iterate through the open files collection
        for ofiles in ofiles_list:

            ## The iterator itself
            for c_count in count():

                ## Fill the buffer with the next set of chunks from the files
                buffer = self.load_chunks( ofiles, c_count )

                ## If the returned buffer is None it means that no more data could be found in the ofiles
                if buffer is None:
                    break

                ## Iterate through the buffer
                for sample in buffer:

                    ## Get the MLP input and target variables
                    inputs = sample[:-2]
                    targets = sample[-2:]
                    yield inputs, targets

        return 0

    def load_chunks(self, files, c_count):

        ## Work out the location of the new chunk
        start = c_count * self.chnk_size
        end = start + self.chnk_size

        ## Get a chunk from each file
        buffer = []
        for f in files:

            ## Running "with" ensures the file is closed
            with h5py.File( f, 'r' ) as hf:
                chunk = hf["data/table"][start:end]
                chunk = np.array([c[1] for c in chunk])

            ## If the chunk is non empty we add it to the buffer
            if len(chunk) != 0:
                buffer.append(chunk)

        ## If the list is empty it means that no files have any data left
        if len(buffer) == 0:
            return None

        ## Get the buffer as a single array and shuffle
        buffer = np.vstack( buffer )
        np.random.shuffle( buffer )
        return buffer

    def shuffle_files(self):
        np.random.shuffle( self.file_list )
