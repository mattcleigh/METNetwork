import torch as T
import torch.nn as nn

from METNetwork.Resources import Layers as myLL

class MET_MLP(nn.Module):
    """ A configurable MLP for MET calculation.
        The network also can have skip arcs
    """

    def __init__( self, name, cut_calo, cut_track, n_in, act, depth, width, skips, nrm, drpt, dev = 'auto' ):
        super(MET_MLP, self).__init__()

        self.cut_calo = cut_calo
        self.cut_track = cut_track

        ## Defining the network features
        self.mlp = myLL.res_mlp_creator( n_in=n_in, n_out=2, depth=depth, width=width,
                                         skips=skips, act_h=act, nrm=nrm, drpt=drpt )

        ## Moving the network to the device
        if dev == 'auto':
            self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        else:
            self.device = T.device(dev)
        self.to(self.device)

        print("\nNetwork structure: {}".format(name))
        print(self)

    def forward( self, inputs ):

        if self.cut_calo:
            inputs[:, 20:25] = 0
            inputs[:, 61:65] = 0

        if self.cut_track:
            inputs[:, 25:29] = 0

        return self.mlp(inputs)
