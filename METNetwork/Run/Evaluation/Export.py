import onnx
import numpy as np
import pandas as pd

import torch as T
import torch.nn as nn

import METNetwork.Resources.Model

class DummyNet(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.lin = nn.Linear(inp, out)

    def forward(self, data):
        return self.lin(data)

class Enveloped_Model(nn.Module):
    def __init__(self, network, stats):
        super().__init__()
        self.network = network
        self.means = stats[0,:]
        self.devs = stats[1,:]

        ## These columns are fixed based on the METNet tool in athena
        self.features = [ 'Tight_Final_ET', 'Tight_Final_EX', 'Tight_Final_EY', 'Tight_Final_SumET', 'Tight_Sig',
                         'Loose_Final_ET', 'Loose_Final_EX', 'Loose_Final_EY', 'Loose_Final_SumET', 'Loose_Sig',
                         'Tghtr_Final_ET', 'Tghtr_Final_EX', 'Tghtr_Final_EY', 'Tghtr_Final_SumET', 'Tghtr_Sig',
                         'FJVT_Final_ET',  'FJVT_Final_EX',  'FJVT_Final_EY',  'FJVT_Final_SumET',  'FJVT_Sig',
                         'Calo_Final_ET',  'Calo_Final_EX',  'Calo_Final_EY',  'Calo_Final_SumET',  'Calo_Sig',
                         'Track_Final_ET', 'Track_Final_EX', 'Track_Final_EY', 'Track_Final_SumET',
                         'Tight_RefJet_ET', 'Tight_RefJet_EX', 'Tight_RefJet_EY', 'Tight_RefJet_SumET',
                         'Loose_RefJet_ET', 'Loose_RefJet_EX', 'Loose_RefJet_EY', 'Loose_RefJet_SumET',
                         'Tghtr_RefJet_ET', 'Tghtr_RefJet_EX', 'Tghtr_RefJet_EY', 'Tghtr_RefJet_SumET',
                         'FJVT_RefJet_ET', 'FJVT_RefJet_EX', 'FJVT_RefJet_EY', 'FJVT_RefJet_SumET',
                         'Tight_Muons_ET', 'Tight_Muons_EX', 'Tight_Muons_EY', 'Tight_Muons_SumET',
                         'Tight_RefEle_ET', 'Tight_RefEle_EX', 'Tight_RefEle_EY', 'Tight_RefEle_SumET',
                         'Tight_RefGamma_ET', 'Tight_RefGamma_EX', 'Tight_RefGamma_EY', 'Tight_RefGamma_SumET',
                         'Loose_PVSoftTrk_ET', 'Loose_PVSoftTrk_EX', 'Loose_PVSoftTrk_EY', 'Loose_PVSoftTrk_SumET',
                         'Calo_SoftClus_ET', 'Calo_SoftClus_EX', 'Calo_SoftClus_EY', 'Calo_SoftClus_SumET',
                         'ActMu', 'NVx_2Tracks', 'NVx_4Tracks', 'PV_NTracks',
                         'N_Muons', 'N_Ele', 'N_Gamma', 'N_Jets', 'N_FWD_Jets',
                         'SumET_FWD_Jets', 'Sum_JetPU', 'Tight_Phi' ]

        ## Get the names and indices of the elements involved with the rotation
        xf_names = [ f for f in self.features if "EX" in f ]
        yf_names = [ f for f in self.features if "EY" in f ]
        self.xs = [ self.features.index(f) for f in xf_names ]
        self.ys = [ self.features.index(f) for f in yf_names ]

    def forward(self, data):
        data, angles = self.pre_process(data)
        data = self.network(data)
        data = self.pst_process(data, angles)
        return data

    def pre_process(self, inpts):

        ## Apply the rotations
        angles = inpts[:, -1:]
        new_x =   inpts[:, self.xs] * T.cos(angles) + inpts[:, self.ys] * T.sin(angles)
        new_y = - inpts[:, self.xs] * T.sin(angles) + inpts[:, self.ys] * T.cos(angles)
        inpts[:, self.xs] = new_x
        inpts[:, self.ys] = new_y

        ## We need to delete the Tight_EX, Tight_Y, Tight_Phi
        inpts = T.cat( [ inpts[:, 0:1], inpts[:, 3:-1] ], dim = 1 )

        ## Apply the standardisation
        inpts = (inpts - self.means[:-3]) / self.devs[:-3]

        return inpts, angles.squeeze()

    def pst_process(self, output, angles):

        ## Undo the standardisation
        output = (output * self.devs[-2:] + self.means[-2:])

        ## Undo the rotations
        new_x =   output[:, 0] * T.cos(-angles) + output[:, 1] * T.sin(-angles)
        new_y = - output[:, 0] * T.sin(-angles) + output[:, 1] * T.cos(-angles)
        output[:, 0] = new_x
        output[:, 1] = new_y

        ## We apply this rotation in batch
        return output

def main():

    ## Load the neural network
    model = Model.METNET_Agent(name = "METNet_PLSWRK_LY_23_05_21_46858343_0", save_dir = "../Saved_Networks/PLSWRK/")
    model.setup_network( act = nn.SiLU(),
                         depth = 5, width = 1024, skips = 0,
                         nrm = True, drpt = 0.0, dev = "cpu" )
    model.load("best", get_opt = False)
    model.network.eval()

    ## Load the stat file for pre-post processing
    stats = np.loadtxt( model.save_dir+model.name+"/stat.csv",
                        skiprows=1,
                        delimiter=",",
                        dtype=np.float32 )

    ## Envelop the model with its pre and post processing steps
    # env_model = Enveloped_Model(model.network, stats)
    dum_model = DummyNet(77, 2)

    ## Produce a simple input to the neural network with the full input!
    dummy_input = T.rand( (5, 77), dtype=T.float32 )
    dynamic_axes = { 'input'  : {0 : 'batch_size' },
                     'output' : {0 : 'batch_size' } }

    T.onnx.export(  dum_model,
                    dummy_input,
                    "dummy_network.onnx",
                    verbose=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                    opset_version=11 )

    ## Load and print the model
    model = onnx.load("dummy_network.onnx")
    print( onnx.helper.printable_graph(model.graph) )
    onnx.checker.check_model(model)

if __name__ == '__main__':
    main()
