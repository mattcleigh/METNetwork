import numpy as np
import geomloss as gl

import torch as T
import torch.nn as nn
import torch.optim as optim

import METNetwork.Resources.Networks as myNW

def mlp_creator( n_in=1, n_out=1, depth=2, width=32,
                 act_h='lrlu', act_o=None, nrm=False, drp=0, widths=[],
                 as_list=False ):

    ## The widths argument overrides width and depth
    if not widths:
        widths = depth * [ width ]

    ## Input block
    blocks = [ myNW.MLPBlock(n_in, widths[0], act_h, nrm, drp) ]

    ## Hidden blocks
    for w1, w2 in zip(widths[:-1], widths[1:]):
        blocks += [ myNW.MLPBlock(w1, w2, act_h, nrm, drp) ]

    ## Output block, optional for creating seperate streams
    if n_out:
        blocks += [ myNW.MLPBlock(widths[-1], n_out, act_o, False, 0) ]

    if as_list: return blocks     ## Return as a list if required
    return nn.Sequential(*blocks) ## Otherwise automatically convert to a pytorch squential object

def get_act(name):
    return {
        'relu': nn.ReLU(),
        'lrlu': nn.LeakyReLU(0.1),
        'silu': nn.SiLU(),
        'selu': nn.SELU(),
        'sigm': nn.Sigmoid(),
        'tanh': nn.Tanh(),
    }[name]

def get_loss(name, **kwargs):
    return {
        'l1loss': nn.L1Loss(reduction='none', **kwargs),
        'l2loss': nn.MSELoss(reduction='none', **kwargs),
        'hbloss': nn.HuberLoss(reduction='none', **kwargs),
        'celoss': nn.CrossEntropyLoss(reduction='none', **kwargs),
        'bcewll': nn.BCEWithLogitsLoss(reduction='none', **kwargs),
        'snkhrn': gl.SamplesLoss('sinkhorn', p=1, blur=0.01),
        'engmmd': gl.SamplesLoss('energy'),
    }[name]

def get_opt(name, params, lr, **kwargs):
    if   name == 'adam': return optim.Adam(params, lr=lr, **kwargs)
    elif name == 'rmsp': return optim.RMSprop(params, lr=lr, **kwargs)
    elif name == 'sgd':  return optim.SGD(params, lr=lr, **kwargs)
    else:
        raise ValueError('No optimiser with name ', name)

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def chunk_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))

def sel_device(dev):
    """
    Returns a pytorch device, includes auto option
    """
    if dev=='auto':
        return T.device('cuda' if T.cuda.is_available() else 'cpu')
    else:
        return T.device(dev)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_np(tensor):
    return tensor.detach().cpu().numpy()

def move_dev(values, dev):
    """
    Moves the values tensor to the targetted device.
    This function calls pytorch's .to() but allows for values to be a
    list of tensors or a tuple of tensors.
    """
    if isinstance(values, tuple):
        return tuple( v.to(dev) for v in values )
    elif isinstance(values, list):
        return [ v.to(dev) for v in values ]
    else:
        return values.to(dev)

def update_avg(old, new, i):
    return old + (new - old) / i

def update_avg_dict(old_dict, new_tup, i):
    """
    Given a dictionary of averages, update the entries using a tuple of new values
    """
    for k, n in zip(old_dict, new_tup):
        old_dict[k] = update_avg(old_dict[k], n, i)

def feature_list():
    return [ 'Tight_Final_ET', 'Tight_Final_EX', 'Tight_Final_EY', 'Tight_Final_SumET', 'Tight_Sig',
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


def setup_input_list(inpt_rmv, do_rot):

    ## Start with the full feature list
    inputs = feature_list()

    ## We always remove the rotation angle as this is used only for pre-post processing
    inputs.remove('Tight_Phi')

    ## We remove the tight ex and ey if doing the rotations
    if do_rot:
        inputs.remove('Tight_Final_EX')
        inputs.remove('Tight_Final_EY')

    ## Cycle through all of the inputs
    for inpt in inputs.copy():

        ## Check against all possible keys
        for key in inpt_rmv.split(','):

            ## Remove the element if it matches a single key
            if key in inpt:
                inputs.remove(inpt)
                break

    ## In the instance that the input list is empty we let one variable through
    ## This is to allow dummy networks to be initialised for testing and plotting
    if not inputs:
        inputs.append('Tight_Final_ET')

    return inputs

class AverageValueMeter:
    '''
    Computes and stores the average, sum, and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
