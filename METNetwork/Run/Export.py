import onnx
import json
import numpy as np
import pandas as pd
import onnxruntime as rt

from datetime import date
from pathlib import Path

import torch as T
import torch.nn as nn

from METNetwork.Resources import Model
import METNetwork.Resources.Utils as myUT
import METNetwork.Resources.Networks as myNW

class DummyNet(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.lin = nn.Linear(inp, out)

    def forward(self, data):
        return self.lin(data)

def addMetaData(name, desc, custom_dict, obj_dict, hyp_dict, inpt_list):
    '''
    This function takes a newly exported model and adds metadata using a given dictionary
    '''

    ## Save the description and dictionary within the .onnx file itself as metadata
    model = onnx.load(name)
    for key, value in dict(**custom_dict, **obj_dict, **hyp_dict).items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value
    model.doc_string = desc
    onnx.save(model, name, desc)

    ## Save the description and dictionary as a seperate text file with the same name
    with open(Path(name).with_suffix('.txt'), 'w') as f:
        print('A metadata file for:', file=f)

        print('->', Path(name).name, ':', desc, '\n', file=f)
        for k, v in custom_dict.items():
            print(k, ':', v, file=f)

        print('\nThe following (baseline) object selections were used to derive the inputs for training.', file=f)
        print('It is highly advised that you stick to these configurations:', file=f)
        for k, v in obj_dict.items():
            print(k, ':', v, file=f)

        print('\nThe network was configured and trained using the following hyper-parameters.', file=f)
        for k, v in hyp_dict.items():
            print(k, ':', v, file=f)

        print('\nThe network uses only the following variables produced by the METNet tool.', file=f)
        for v in inpt_list:
            print(v, file=f)

def main():

    ## The input and output file names
    net_folder = '/mnt/scratch/Saved_Networks/Presentation'
    net_name = '49484133_5_18_08_21'
    output_name = 'RotatedMagIndep_v0.onnx'

    ## Load the trained network and the model for its dictionary
    model = Model.METNET_Agent(net_name, net_folder)
    model.load(dict_only=True)
    net = T.load(Path(net_folder, net_name, 'models/net_best'))
    print(net)
    
    ## Configure the network for full pass evaluation
    net.to('cpu')
    net.eval()
    net.do_proc = True

    # desc = 'A placeholder ONNX model converted from Pytorch, should be used for testing purposes only!'
    desc = 'A trained MLP for missing transverse momentum reconstruction converted from Pytorch'

    custom_dict = {
        'author':'Matthew Leigh, University of Geneva, matthew.leigh@cern.ch',
        'date': date.today().strftime("%d/%m/%y"),
        'trained_at':'Unversity of Geneva High Performance Computing Center (Baobab)',
        'training_set':'mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYS.e6337_s3126_r10724_p4355',
        'AthenaRelease':'AthAnalysis 21.2.170',
        'PytorchVersion':T.__version__,
        'OnnxVersion':onnx.__version__,
    }

    obj_dict = {
        'Ele.Pt':'10 GeV',
        'Ele.Eta':'2.47',
        'Ele.Id':'LooseAndBLayerLLH',
        'Ele.CrackVeto':'False',
        'Ele.z0':'0.5',
        'Muon.Pt':'10 GeV',
        'Muon.Eta':'2.7',
        'Muon.Id':'Medium',
        'Muon.z0':'0.5',
        'Photon.Pt':'25 GeV',
        'Photon.Eta':'2.37',
        'Photon.Id':'Tight',
        'Jet.Pt':'20 GeV'
    }

    hyp_dict = model.get_dict()
    inpt_list = np.array(myUT.feature_list())[net.inpt_idxes.tolist()]

    ###################

    full_name = '../Output/' + output_name

    ## Export the model
    T.onnx.export( net,                                  ## The torch model
                   T.rand((5, 77), dtype=T.float32),     ## A dummy input with the correct size for tracing
                   full_name,                            ## The output name
                   input_names=["input"],                ## The input collection name  ('input'  required by AthAlg tool!)
                   output_names=["output"],              ## The output collection name ('output' required by AthAlg tool!)
                   dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}}, ## Required for batch_propagation
                   opset_version=11 )                    ## Leave as 11, seems to work fine

    dd = onnx.load(full_name)
    print( onnx.helper.printable_graph(dd.graph) )

    ## Add the dictionary to the new model
    addMetaData(full_name, desc, custom_dict, obj_dict, hyp_dict, inpt_list)

if __name__ == '__main__':
    main()
