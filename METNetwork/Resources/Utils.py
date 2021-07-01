import numpy as np
import geomloss as gl

import torch as T
import torch.nn as nn
import torch.optim as optim

import METNetwork.Resources.Modules as myML

def mlp_creator( n_in=1, n_out=1, depth=2, width=32,
                 act_h='lrlu', act_o=None, nrm=False, drp=0, widths=[],
                 as_list=False ):

    ## The widths argument overrides width and depth
    if not widths:
        widths = depth * [ width ]

    ## Input block
    blocks = [ myML.MLPBlock(n_in, widths[0], act_h, nrm, drp) ]

    ## Hidden blocks
    for w1, w2 in zip(widths[:-1], widths[1:]):
        blocks += [ myML.MLPBlock(w1, w2, act_h, nrm, drp) ]

    ## Output block, optional for creating seperate streams
    if n_out:
        blocks += [ myML.MLPBlock(widths[-1], n_out, act_o, False, 0) ]

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
        'snkhrn': gl.SamplesLoss('sinkhorn')
    }[name]

def get_opt(name, params, lr, **kwargs):
    if   name == 'adam': return optim.Adam(params, lr=lr, **kwargs)
    elif name == 'rmsp': return optim.RMSprop(params, lr=lr, **kwargs)
    elif name == 'sgd':  return optim.SGD(params, lr=lr, **kwargs)

def update_avg(old, new, i):
    return old + (new - old) / i

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
