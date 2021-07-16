import torch as T
import torch.nn as nn

import METNetwork.Resources.Utils as myUT

class MLPBlock(nn.Module):
    def __init__(self, n_in, n_out, act, nrm, drp):
        super().__init__()
        block = [ nn.Linear(n_in, n_out) ]
        if act: block += [ myUT.get_act(act) ]
        if nrm: block += [ nn.LayerNorm(n_out) ]
        if drp: block += [ nn.Dropout(drp) ]
        self.block = nn.Sequential(*block)

    def forward(self, data):
        return self.block(data)

class METNetwork(nn.Module):
    '''
    A network for missing transverse momentum reconstruction. At it's core is a simple and configurable
    multi-layer perceptron. The MLP is enveloped in pre- and post-processing layers, which perform
     - masking
     - scaling
     - rotations
    These outer layers are disabled during training as our trianing datasets
    are already processed!
    The network stores:
     - A list of indices to immediately shrink the input list
     - All the indices of the x and y components in the input list
     - Stats for the input pre-processing
     - Stats for the output post-processing
    '''
    def __init__(self, inpt_list, **mlp_kwargs):
        super().__init__()

        self.do_proc = False
        self.mlp = myUT.mlp_creator(n_in=len(inpt_list), **mlp_kwargs)

        ## Save a mask showing which of the tool's variables are actually used in the neural network
        inpt_idxes = T.tensor( [ myUT.feature_list().index(i) for i in inpt_list ], dtype=T.long)
        self.register_buffer('inpt_idxes', inpt_idxes)

        ## Register a stats buffer with zero's for now, they will be filled if we load a previous save or a training dataset
        inp_stats = T.zeros((2, len(myUT.feature_list())), dtype=T.float32)
        trg_stats = T.zeros((2, 2), dtype=T.float32)
        self.register_buffer('inp_stats', inp_stats)
        self.register_buffer('trg_stats', trg_stats)

        ## Get the names and indices of the elements involved with the rotation
        x_idxes = T.tensor([ inpt_list.index(f) for f in inpt_list if "EX" in f ], dtype=T.long)
        y_idxes = T.tensor([ inpt_list.index(f) for f in inpt_list if "EY" in f ], dtype=T.long)
        self.register_buffer('x_idxes', x_idxes)
        self.register_buffer('y_idxes', y_idxes)

    def set_statistics(self, stats):
        '''
        Use the stats file produced when creating the HDF training datasets to update the onboard
        statistics tensors
        '''
        self.inp_stats = stats[:, :len(myUT.feature_list()) ][:, self.inpt_idxes]
        self.trg_stats = stats[:, -2:]

    def forward(self, data):

        if self.do_proc:
            data, angles = self.pre_process(data)

        data = self.mlp(data)

        if self.do_proc:
            data = self.pst_process(data, angles)

        return data

    def pre_process(self, inpts):
        '''
        Preprocessing is the first step of a full pass using the METNet tool. Which means that the inpts
        must be the raw 77 variables produced by the tool!
        '''

        ## Extract the angle of rotation
        angles = inpts[:, -1:]

        ## Apply the mask to the inpts
        inpts = inpts[:, self.inpt_idxes]

        ## Apply the rotations
        new_x =   inpts[:, self.x_idxes] * T.cos(angles) + inpts[:, self.y_idxes] * T.sin(angles)
        new_y = - inpts[:, self.x_idxes] * T.sin(angles) + inpts[:, self.y_idxes] * T.cos(angles)
        inpts[:, self.x_idxes] = new_x
        inpts[:, self.y_idxes] = new_y

        ## Apply the standardisation
        inpts = (inpts - self.inp_stats[0]) / self.inp_stats[1]

        return inpts, angles.squeeze()

    def pst_process(self, output, angles):

        ## Undo the standardisation
        output = output * self.trg_stats[1] + self.trg_stats[0]

        ## Undo the rotations
        new_x =   output[:, 0] * T.cos(-angles) + output[:, 1] * T.sin(-angles)
        new_y = - output[:, 0] * T.sin(-angles) + output[:, 1] * T.cos(-angles)
        output[:, 0] = new_x
        output[:, 1] = new_y

        return output
