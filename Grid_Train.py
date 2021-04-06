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

    parser.add_argument( "--net_dir",
                         type = str,
                         help = "The output folder containing the saved networks",
                         required = True )

    parser.add_argument( "--do_rot",
                         type = str2bool,
                         help = "Whether to use a pre-rotated dataset",
                         required = True )

    parser.add_argument( "--weight_to",
                         type = float,
                         help = "The location of the falling edge of the plateau in MeV",
                         required = True )

    parser.add_argument( "--bsize",
                         type = int,
                         help = "The batch size",
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

    parser.add_argument( "--lr",
                         type = float,
                         help = "The optimiser learning rate / step size",
                         required = True )

    return parser.parse_args()

def print_args( args ):
    print("\nRunning job with options:")
    for key, value in vars(args).items():
        print( " - {:10}: {}".format(key, value))
    print("")

def pass_blacklist( args ):

    blacklist = [
                    # ( "depth", 5,   "width", 256 ),
                    # ( "skips", 0,   "width", 256 ),
                    # ( "nrm", False, "lr", 1e-3 ),
    ]

    for a1, v1, a2, v2 in blacklist:
        if getattr(args, a1) == v1 and getattr(args, a2) == v2:
            print( "Argument combination is on the blacklist!" )
            print( "---> ( {} = {} ) and ( {} = {} )".format(a1, v1, a2, v2) )
            return False
    return True

# python Grid_Train.py --data_dir ../Data/ --net_dir Saved_Models --name GridTest --do_rot True --weight_to 2e5 --bsize 512 --depth 9 --width 256 --skips 2 --nrm True --lr 1e-3

def main():

    ## Get and print the arguments
    args = get_args()
    print_args( args )

    ## Discard blacklisted argument matches
    if not pass_blacklist( args ):
        return 0

    ## Initialise the model
    model = Model.METNET_Agent( name = args.name, save_dir = args.net_dir )

    ## Load up the dataset
    model.setup_dataset( data_dir   = args.data_dir,
                         do_rot     = args.do_rot,
                         weight_to  = args.weight_to,
                         valid_frac = 1e-1,
                         n_ofiles   = 6, chnk_size = 2048,
                         batch_size = args.bsize, n_workers = 4 )

    ## Initialise the prepost-MLP network
    model.setup_network( act = nn.LeakyReLU(0.1),
                         depth = args.depth, width = args.width, skips = args.skips,
                         nrm = args.nrm, drpt = 0.0 )

    ## Setup up the parameters for training
    model.setup_training( loss_fn = nn.SmoothL1Loss(),
                          lr = args.lr,
                          clip_grad = 0 )

    ## Run the training loop
    model.run_training_loop( max_epochs = 1000, patience = 10, sv_every = 100 )

if __name__ == '__main__':
    main()
