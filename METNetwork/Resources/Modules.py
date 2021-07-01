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
