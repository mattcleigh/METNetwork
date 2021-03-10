import sys
home_env = '../'
sys.path.append(home_env)

import glob
import h5py
import linecache
import numpy as np
import numpy.random as rd

import torch as T
import torchvision as TV
from torch.utils.data import Dataset

def unorm_batch( batch, means, devs ):
    return T.mul( batch, devs ) + means

def norm_batch( batch, means, devs ):
    return T.div( ( batch - means ), devs+1e-6 )

def rotate_batch( batch, angles, xs ):
    ys = xs + 1
    rotated_x =   batch.iloc[:, xs] * np.cos(angles) + batch.iloc[:, ys] * np.sin(angles)
    rotated_y = - batch.iloc[:, xs] * np.sin(angles) + batch.iloc[:, ys] * np.cos(angles)
    batch.iloc[:, xs] = rotated_x
    batch.iloc[:, xs] = rotated_y

# def Concat_MET_Dataset( search_tag ):
#
#     ## Find all the file paths in the data directory
#     file_list = glob.glob( search_tag )
#     n_files = len(self.file_list)
#
#     data_list = []
#     for file in file_list:
#         data_list.append( myDS.METDataset( file ) )
#     ConcatDataset( [ myDS.METDataset( file, stat_file=stat_file, x_ids=x_ids ) for file in self.file_list ] )
#
# class METDataset(Dataset):
#     def __init__( self, file_name, full_return ):
#
#         self.file_name = file_name
#         self.full_return = full_return
#         self.n_samples = len(h5py.File( self.file_name, 'r' )["data"])
#
#     def __len__(self):
#         return self.n_samples
#
#     def __getitem__(self, idx):
#
#         ##
#         file_data = h5py.File( self.file_name, 'r' )["data"]
#         file_data = h5py.File( 'output-0.h5', 'r' )["data"]["table"]
#         sample = file_data[i][1]
#         ## Convert the line to a torch tensor and extract the angle
#         sample = T.from_numpy(file_data[idx])
#         angle = sample[74]
#
#         ## We return the whole transformed sample if desired (needed for generating stats)
#         if self.full_return:
#             return sample
#
#         ## Get the MLP input (without angle) and target variables
#         inputs = sample[:74]
#         targets = sample[75:77]
#
#         return inputs, targets
