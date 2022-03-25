import h5py
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from itertools import count

import torch as T
from torch.utils.data import IterableDataset

from METNetwork.resources.feature_list import feature_list
from METNetwork.resources.sampler import Sampler

from mattstools.utils import chunk_given_size

## The size of the HDF chunks prepared by create_HDF.py
CHUNK_SIZE = 399


def create_input_list(inpt_rmv: str, do_rot: bool) -> list:
    """Return a list of inputs to use in the network

    args:
        inpt_rmv: A string of comma seperated searches indicating which vars to remove
        do_rot: If the data being read will be rotated or not
    """

    ## Start with the full feature list
    inputs = feature_list()

    ## Remove the rotation angle as this is used only for pre-post processing
    inputs.remove("Tight_Phi")

    ## Remove the Tight x and y componets if doing rotations
    if do_rot:
        inputs.remove("Tight_Final_EX")  ## Should be equal to Tight_Final_ET
        inputs.remove("Tight_Final_EY")  ## Should be 0

    ## Create a list of inputs to remove from the original string
    remove_keys = inpt_rmv.split(",")

    ## Cycle through all of the inputs and remove if it matches a key
    sel_inpts = []
    for inpt in inputs:

        ## Outlier behaviour: If do rot and rmv contains _ET, nothing of Tight will
        ## Remain in the input list! So we must test agains other conditions
        if inpt == "Tight_Final_ET" and "_ET" in remove_keys:
            rmvkeys = remove_keys.copy()
            rmvkeys.remove("_ET")
            if not any([k in inpt for k in rmvkeys]):
                sel_inpts.append(inpt)

        ## Otherwise just use the full list of keys
        elif not any([k in inpt for k in remove_keys]):
            sel_inpts.append(inpt)

    ## In the instance that the input list is now empty we let one variable through
    ## This is to allow dummy networks to be initialised for testing and plotting
    if not sel_inpts:
        sel_inpts.append("Tight_Final_ET")

    return sel_inpts


class StreamMETDataset(IterableDataset):
    """Iterable dataset class for METNet streaming in HDF file inputs
    - Yeilds not only the inputs and targets, but also a sample weight
    - Yeilds results in batches for speed

    Works as follows with multitreading for each epoch:
    - Thread is assigned a portion of HDF files (exclusive)
    - Thread groups files into mini collections it will have open at a time
    - Loops through mini collections
      - Reads a chuck of data from each into a buffer which is shuffled
      - Calculates sample weights for whole buffer
      - Loops through samples in buffer yeilding samples and weights

    Will end iteration of chunks when all open files are finished
    Will end iteration of epoch when worker has returned its share of samples even
    if that means the files are run through again

    Amount of samples stored in memory at a given time is:
    - n_threads x n_ofiles x chunk_size
    """

    def __init__(
        self,
        dset: str = "train",
        path: str = "SetMePlease",
        do_rot: bool = "False",
        inpts_rmv: str = "xxx",
        n_ofiles: int = 64,
        chunk_size: int = CHUNK_SIZE * 10,
        do_single_chunk: bool = False,
        sampler_kwargs: dict = None,
    ):
        """
        args:
            dset: Which dataset to pull from, either train or val
            path: The location of the datafiles
            do_rot: If the rotated or the raw data should be loaded
            inpts_rmv: Which inputs variables should be removed / ignored
            n_ofiles:  Nnumber of files to read from simultaneously
                       larger = more memory but better shuffling
            chunk_size: The size of the chunk to read from each of the ofiles
            do_single_chunk: Only load one chunk per HDF file
            sampler_kwargs: Keyword arguments for the weighted sampler
        """
        super().__init__()

        print(f"\nCreating dataset: {dset}")

        ## Check dset type
        if dset not in ["train", "val"]:
            raise ValueError(f"Unrecognized dset: {dset}")

        ## Default dict
        sampler_kwargs = sampler_kwargs or {}

        ## Class attributes
        self.do_rot = do_rot
        self.n_ofiles = n_ofiles

        self.weight_exist = sampler_kwargs["weight_to"] > 0  ## Kept constant
        self.do_weights = self.weight_exist  ##  Can be toggled on and off

        ## Save the input list and the var list which always has these three extra!
        self.inpt_list = create_input_list(inpts_rmv, do_rot)
        self.var_list = self.inpt_list + ["True_ET", "True_EX", "True_EY"]

        ## Get the list of files to use
        self.path = Path(path, "Rotated" if do_rot else "Raw")
        self.f_list = list(self.path.glob("*.h5"))

        ## Exit if no files can be found
        if not self.f_list:
            raise LookupError("No HDF files could be found in ", path)

        ## Init the sampler weight class
        if self.weight_exist:
            self.sampler = Sampler(self.path, **sampler_kwargs)

        ## For validation we only use the first chunk of 399 in the file
        self.is_val = dset == "val"
        if self.is_val:
            self.abs_start = 0
            self.chunk_size = CHUNK_SIZE
            self.do_single_chunk = True

        ## For training we can use all as many chunks as desired
        else:
            self.abs_start = CHUNK_SIZE
            self.chunk_size = chunk_size
            self.do_single_chunk = do_single_chunk

        ## Count the number of samples used
        if self.do_single_chunk:
            self.n_samples = self.chunk_size * len(self.f_list)
        else:
            self.n_samples = 0
            for file in tqdm(self.f_list, desc="counting events"):
                with h5py.File(file, "r") as hf:
                    self.n_samples += len(hf["data/table"])
            self.n_samples -= CHUNK_SIZE * len(self.f_list) ## Subtract val samples

    def get_preprocess_info(self):
        """Return a dictionary of pre_processing and stat information"""
        info = {}

        ## Save a mask corresponding to the input list to pull from full feature list
        info["inpt_idxes"] = T.tensor(
            [feature_list().index(i) for i in self.inpt_list], dtype=T.long
        )

        ## Get the names and indices of the selected vars involved with the rotation
        info["x_idxes"] = T.tensor(
            [self.inpt_list.index(f) for f in self.inpt_list if "EX" in f], dtype=T.long
        )
        info["y_idxes"] = T.tensor(
            [self.inpt_list.index(f) for f in self.inpt_list if "EY" in f], dtype=T.long
        )

        ## Get the dataset means and devs
        stats = T.tensor(
            pd.read_csv(Path(self.path, "stats.csv")).to_numpy(), dtype=T.float32
        )
        info["inpt_means"] = stats[0, info["inpt_idxes"]]
        info["inpt_sdevs"] = stats[1, info["inpt_idxes"]]
        info["outp_means"] = stats[0, -2:]
        info["outp_sdevs"] = stats[1, -2:]

        return info

    def shuffle_files(self):
        """Shuffles the file list so that each worker will get a different subset
        - Should call inbetween each epoch
        """
        np.random.shuffle(self.f_list)

    def weight_on(self):
        """Turns on calculating per sample weights
        - Needed for the training and validation epochs
        """
        self.do_weights = self.weight_exist

    def weight_off(self):
        """Turns off calculating per sample weights
        - Needed for the performance evaluation steps
        """
        self.do_weights = False

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        """
        Called automatically whenever an iterator is created on the
        dataloader responsible for this dataset set.
        ie: Whenever 'for batch in dataloader: ...' is executed

        This function is called SEPARATELY for each thread
        Think of it as a worker (thread) initialise function
        """

        ## Get the worker info
        w_info = T.utils.data.get_worker_info()

        ## If it is None we are doing single process loading, worker uses whole f_list
        if w_info is None:
            is_main_wrkr = True
            worker_files = self.f_list
            worker_samples = self.n_samples

        ## For multiple workers break up the file list so they each work on a subset
        else:
            is_main_wrkr = w_info.id == 0
            worker_files = np.array_split(self.f_list, w_info.num_workers)[w_info.id]
            if self.do_single_chunk:
                worker_samples = self.chunk_size * len(worker_files)
            else:
                worker_samples = self.n_samples // w_info.num_workers


        ## Cycle until stop iteration criterion is met
        n_returned = 0
        while True:

            ## Partition the worker's file list into ones it will have open at a time
            np.random.shuffle(worker_files)
            ofiles_list = chunk_given_size(worker_files, self.n_ofiles)

            ## First iterate through the open files collection
            for ofiles in ofiles_list:

                ## Then iterate through the chunks taken from each of the files
                for c_count in count():

                    ## Fill the buffer with the next set of chunks from the files
                    buffer = self.load_chunks(ofiles, c_count)

                    ## If the returned buffer is empty there are no more events!
                    if not buffer.size:
                        break

                    ## Calculate all weights for the buffer using truth vars (last 3)
                    if self.do_weights:
                        weights = self.sampler.apply(buffer[:, -3:])

                        ## Drop all samples with weights of zero for the sampling meth
                        if self.sampler.weight_ratio > 0:
                            buffer = buffer[weights > 0]
                            weights = weights[weights > 0]
                    else:
                        weights = np.ones(len(buffer))

                    ## Split the inputs and target matrices
                    inputs, targets = buffer[:, :-3], buffer[:, -2:]

                    ## Iterate through the samples taken from the buffer
                    for input, target, weight in zip(inputs, targets, weights):
                        yield input, target, weight

                        ## Look for potential exit condition
                        n_returned += 1
                        if n_returned >= worker_samples:

                            ## Main worker is responsible for shuffling upon exit
                            if is_main_wrkr:
                                self.shuffle_files()

                            return

                    ## Break after a single chunk
                    if self.do_single_chunk:
                        break

    def load_chunks(self, files: list, c_count: int):
        """Returns a buffer of samples built from one chunk taken from a collection
        of files

        args:
            files: List of HDF files to load data from
            c_count: Index of current buffer to load
        """

        ## Work out the bounds of the new chunk within the file
        if self.chunk_size == "all":
            start = None
            stop = None
        else:
            start = c_count * self.chunk_size + self.abs_start
            stop = start + self.chunk_size + self.abs_start

        buffer = []
        for f in files:
            with h5py.File(f, "r") as hf:

                ## Loading data with the var list gives an array of tuples, must conv
                chunk = hf["data/table"][start:stop][self.var_list]
                buffer += list(map(list, chunk))  ## Quicker than [list(e) for ...]

        ## Shuffle and return the buffer as a numpy array
        np.random.shuffle(buffer)
        return np.array(buffer, dtype=np.float32)
