from Resources import Model
import argparse
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

    parser.add_argument( "--do_rot",
                         type = str2bool,
                         help = "Whether to use a pre-rotated dataset",
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

def main():

    args = get_args()

    ## Initialise the model
    model = Model.METNET_Agent( name = args.name, save_dir = "Saved_Models" )

    ## Load up the dataset
    model.setup_dataset( data_dir   = "/mnt/scratch/Data/",
                         do_rot     = args.do_rot,
                         valid_frac = 5e-2,
                         n_ofiles   = 32, chnk_size = 2048,
                         batch_size = bsize, n_workers = 2 )

    ## Initialise the prepost-MLP network
    model.setup_network( act = nn.LeakyReLU(0.1),
                         depth = args.depth, width = args.width, skips = args.skips,
                         nrm = args.nrm, drpt = 0.0 )

    ## Setup up the parameters for training
    model.setup_training( loss_fn = nn.SmoothL1Loss(),
                          lr = args.lr,
                          clip_grad = 0 )

    ## Run the training loop
    model.run_training_loop( max_epochs = 1, patience = 5, sv_every = 5 )

    ## Save some prformance metrics using the best version of the network
    model.save_best_perf()

if __name__ == '__main__':
    main()
