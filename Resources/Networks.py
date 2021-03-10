import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Layers as myLL

import torch as T
import torch.nn as nn


class MET_MLP(nn.Module):
    """ A configurable MLP for MET calculation.
        The network also can have skip arcs
    """

    def __init__( self, name, act, depth, width, skips, nrm, drpt ):
        super(MET_MLP, self).__init__()

        ## Defining the network features
        self.__dict__.update(locals())
        self.mlp = myLL.res_mlp_creator( n_in=74, n_out=2, depth=depth, width=width,
                                         skips=skips, act_h=act, nrm=nrm, drpt=drpt )

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        print("\nNetwork structure: {}".format(self.name))
        print(self)

    def forward( self, inputs ):
        return self.mlp(inputs)
