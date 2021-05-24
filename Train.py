import argparse
from Resources import Model
import torch.nn as nn

def str2bool(v):
    if isinstance(v, bool): return v
    if   v.lower() in ("yes", "true", "t", "y", "1"): return True
    elif v.lower() in ("no", "false", "f", "n", "0"): return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--name",
                         type = str,
                         help = "The name to use for saving the network",
                         required = True )

    parser.add_argument( "--data_dir",
                         type = str,
                         help = "The folder containing the Raw and Rotated datasets",
                         required = True )

    parser.add_argument( "--save_dir",
                         type = str,
                         help = "The output folder containing the saved networks",
                         required = True )

    parser.add_argument( "--do_rot",
                         type = str2bool,
                         help = "Whether to use a pre-rotated dataset",
                         required = True )

    parser.add_argument( "--weight_to",
                         type = float,
                         help = "The location of the falling edge of the plateau in GeV",
                         required = True )

    parser.add_argument( "--weight_ratio",
                         type = float,
                         help = "The maximum allowed loss weight ratio between two events",
                         required = True )

    parser.add_argument( "--weight_shift",
                         type = float,
                         help = "The gradient [-1, 1] of the linear shift applied to the event weights",
                         required = True )

    parser.add_argument( "--v_frac",
                         type = float,
                         help = "The fraction of input files reserved for the validation set",
                         required = True )

    parser.add_argument( "--n_ofiles",
                         type = int,
                         help = "The number of files that are opened together for a single buffer",
                         required = True )

    parser.add_argument( "--chnk_size",
                         type = int,
                         help = "The size of the chunks taken from each of the ofiles to fill the buffer",
                         required = True )

    parser.add_argument( "--bsize",
                         type = int,
                         help = "The batch size",
                         required = True )

    parser.add_argument( "--n_workers",
                         type = int,
                         help = "The number of worker threads to prepare the batches",
                         required = True )

    parser.add_argument( "--depth",
                         type = int,
                         help = "The number of hidden layers in the MLP",
                         required = True )

    parser.add_argument( "--width",
                         type = int,
                         help = "The number of neurons per hidden layer in the MLP",
                         required = True )

    parser.add_argument( "--skips",
                         type = int,
                         help = "The number of hidden layers skipped per residual arc (0 for no arcs)",
                         required = True )

    parser.add_argument( "--nrm",
                         type = str2bool,
                         help = "Whether to do batch norm in each hidden layer",
                         required = True )

    parser.add_argument( "--drpt",
                         type = float,
                         help = "The drop-out probability for each hidden layer",
                         required = True )

    parser.add_argument( "--lr",
                         type = float,
                         help = "The optimiser learning rate / step size",
                         required = True )

    parser.add_argument( "--grad_clip",
                         type = float,
                         help = "Maximum value of the batch gradient norm",
                         required = True )

    parser.add_argument( "--skn_weight",
                         type = float,
                         help = "The relative weight of the Sinkhorn loss",
                         required = True )

    return parser.parse_args()

def print_args( args ):
    print("\nRunning job with options:")
    for key, value in vars(args).items():
        print( " - {:12}: {}".format(key, value))
    print("")

def pass_blacklist( args ):

    blacklist = [
                    # ( "depth", 9,   "width", 1024 ),
                    # ( "depth", 5,   "width", 256 ),
                    # ( "skips", 0,   "width", 256 ),
    ]

    for a1, v1, a2, v2 in blacklist:
        if getattr(args, a1) == v1 and getattr(args, a2) == v2:
            print( "Argument combination is on the blacklist!" )
            print( "---> ( {} = {} ) and ( {} = {} )".format(a1, v1, a2, v2) )
            return False
    return True

def main():

    ## Get and print the arguments
    args = get_args()
    print_args( args )

    ## Discard blacklisted argument matches
    if not pass_blacklist( args ):
        return 0

    ## Initialise the model
    model = Model.METNET_Agent( name = args.name, save_dir = args.save_dir )

    ## Load up the dataset
    model.setup_dataset( data_dir   = args.data_dir,
                         do_rot     = args.do_rot,
                         weight_to  = args.weight_to, weight_ratio = args.weight_ratio, weight_shift = args.weight_shift,
                         v_frac     = args.v_frac, n_ofiles = args.n_ofiles, chnk_size = args.chnk_size,
                         batch_size = args.bsize, n_workers = args.n_workers )

    ## Initialise the network
    model.setup_network( act = nn.SiLU(),
                         depth = args.depth, width = args.width, skips = args.skips,
                         nrm = args.nrm, drpt = args.drpt )

    ## Setup up the parameters for training
    model.setup_training( loss_fn = nn.SmoothL1Loss( reduction="none" ),
                          lr = args.lr,
                          grad_clip = args.grad_clip,
                          skn_weight = args.skn_weight )

    ## Run the training loop
    model.run_training_loop( max_epochs = 1000, patience = 20, sv_every = 1 )

if __name__ == '__main__':
    main()
