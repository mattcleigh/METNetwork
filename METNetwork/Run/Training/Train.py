import argparse
import torch.nn as nn

import METNetwork.Resources.Utils as myUT

from METNetwork.Resources import Model

def str2bool(v):
    if isinstance(v, bool): return v
    elif v.lower() in ("yes", "true", "t", "y", "1"): return True
    elif v.lower() in ("no", "false", "f", "n", "0"): return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--name",
                         type = str,
                         help = "The name to use for saving the network",
                         required = True )

    parser.add_argument( "--save_dir",
                         type = str,
                         help = "The output folder containing the saved networks",
                         required = True )

    parser.add_argument( "--data_dir",
                         type = str,
                         help = "The folder containing the Raw and Rotated datasets",
                         required = True )

    parser.add_argument( "--do_rot",
                         type = str2bool,
                         help = "To use Tight rotated data or not",
                         required = True )

    parser.add_argument( "--v_frac",
                         type = float,
                         help = "The fraction of input files reserved for the validation set",
                         required = True )

    parser.add_argument( "--inpt_rmv",
                         type = str,
                         help = "A comma seperated list containing the keys to remove from the input list",
                         required = True )

    parser.add_argument( "--n_ofiles",
                         type = int,
                         help = "The number of files that are opened together for a single buffer",
                         required = True )

    parser.add_argument( "--chnk_size",
                         type = int,
                         help = "The size of the chunks taken from each of the ofiles to fill the buffer",
                         required = True )

    parser.add_argument( "--b_size",
                         type = int,
                         help = "The batch size",
                         required = True )

    parser.add_argument( "--n_workers",
                         type = int,
                         help = "The number of worker threads to prepare the batches",
                         required = True )

    parser.add_argument( "--weight_type",
                         type = str,
                         help = "Derive weights based on the 2D target histogram 'trg' or the 1D magnitude histogram 'mag'",
                         required = True )

    parser.add_argument( "--weight_to",
                         type = float,
                         help = "The location of the falling edge of the plateau in GeV",
                         required = True )

    parser.add_argument( "--weight_shift",
                         type = float,
                         help = "The gradient [-1, 1] of the linear shift applied to the event weights",
                         required = True )

    parser.add_argument( "--weight_ratio",
                         type = float,
                         help = "The maximum allowed loss weight ratio between two events, defines sampling vs weighting",
                         required = True )

    parser.add_argument( "--act",
                         type = str,
                         help = "The activation function to use in the MLP",
                         required = True )

    parser.add_argument( "--depth",
                         type = int,
                         help = "The number of hidden layers in the MLP",
                         required = True )

    parser.add_argument( "--width",
                         type = int,
                         help = "The number of neurons per hidden layer in the MLP",
                         required = True )

    parser.add_argument( "--nrm",
                         type = str2bool,
                         help = "Whether to do batch norm in each hidden layer",
                         required = True )

    parser.add_argument( "--drpt",
                         type = float,
                         help = "The drop-out probability for each hidden layer",
                         required = True )

    parser.add_argument( "--opt_nm",
                         type = str,
                         help = "The name of the optimiser to use",
                         required = True )

    parser.add_argument( "--lr",
                         type = float,
                         help = "The optimiser learning rate / step size",
                         required = True )

    parser.add_argument( "--patience",
                         type = int,
                         help = "How many bad epochs will the trainer allow before early stopping",
                         required = True )

    parser.add_argument( "--reg_loss_nm",
                         type = str,
                         help = "The name of the loss function to use for regression",
                         required = True )

    parser.add_argument( "--dst_loss_nm",
                         type = str,
                         help = "The name of the loss function to use for distribution matching",
                         required = True )

    parser.add_argument( "--dst_weight",
                         type = float,
                         help = "The relative weight of the distribution matching loss",
                         required = True )

    parser.add_argument( "--grad_clip",
                         type = float,
                         help = "Maximum value of the batch gradient norm",
                         required = True )

    return parser.parse_args()

def print_args(args):
    print("\nRunning job with options:")
    for key, value in vars(args).items():
        print( " - {:12}: {}".format(key, value))
    print("")

def pass_blacklist(args):

    blacklist = [
                    ( "weight_to", 0,   "weight_type", 'trg' ),
    ]

    for a1, v1, a2, v2 in blacklist:
        if getattr(args, a1) == v1 and getattr(args, a2) == v2:
            raise ValueError('Argument combination is on the blacklist!\n \
                              ---> ( {} = {} ) and ( {} = {} )'.format(a1, v1, a2, v2))

def main():

    ## Collect, print and check the arguments
    args = get_args()
    print_args(args)
    pass_blacklist(args)

    ## Initialise the model
    model = Model.METNET_Agent(args.name, args.save_dir)

    ## Initialise the network
    model.setup_network(args.do_rot, args.inpt_rmv, args.act, args.depth, args.width, args.nrm, args.drpt)

    ## Load up the dataset
    model.setup_dataset(args.data_dir, args.v_frac,
                        args.n_ofiles, args.chnk_size,
                        args.b_size, args.n_workers,
                        args.weight_type, args.weight_to,
                        args.weight_shift, args.weight_ratio)

    ## Setup up the parameters for training
    model.setup_training(args.opt_nm, args.lr,
                         args.patience, args.reg_loss_nm,
                         args.dst_loss_nm, args.dst_weight, args.grad_clip)

    ## Run the training loop
    model.run_training_loop()

if __name__ == '__main__':
    main()
