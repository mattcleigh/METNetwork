import argparse
import torch.nn as nn

from METNetwork.Resources import Model

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

    parser.add_argument( "--save_dir",
                         type = str,
                         help = "The output folder containing the saved networks",
                         required = True )

    parser.add_argument( "--data_dir",
                         type = str,
                         help = "The folder containing the Raw and Rotated datasets",
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

    parser.add_argument( "--b_size",
                         type = int,
                         help = "The batch size",
                         required = True )

    parser.add_argument( "--n_workers",
                         type = int,
                         help = "The number of worker threads to prepare the batches",
                         required = True )

    return parser.parse_args()

def print_args(args):
    print("\nRunning job with options:")
    for key, value in vars(args).items():
        print( " - {:12}: {}".format(key, value))
    print("")

def pass_blacklist(args):

    blacklist = [
                    ( "weight_to", 0,   "weight_ratio", 0.1 ),
                    # ( "depth", 5,   "width", 256 ),
    ]

    for a1, v1, a2, v2 in blacklist:
        if getattr(args, a1) == v1 and getattr(args, a2) == v2:
            raise ValueError('Argument combination is on the blacklist!\n \
                              ---> ( {} = {} ) and ( {} = {} )'.format(a1, v1, a2, v2))

def main():

    ## Get and print the arguments
    args = get_args()
    print_args(args)

    ## Discard blacklisted argument matches
    pass_blacklist(args)

    ## The input list
    inpt_list = [
        'Tight_Final_ET', 'Tight_Final_SumET', 'Tight_Sig',
        'Loose_Final_ET', 'Loose_Final_EX', 'Loose_Final_EY', 'Loose_Final_SumET', 'Loose_Sig',
        'Tghtr_Final_ET', 'Tghtr_Final_EX', 'Tghtr_Final_EY', 'Tghtr_Final_SumET', 'Tghtr_Sig',
        'FJVT_Final_ET',  'FJVT_Final_EX',  'FJVT_Final_EY',  'FJVT_Final_SumET',  'FJVT_Sig',
        'Calo_Final_ET',  'Calo_Final_EX',  'Calo_Final_EY',  'Calo_Final_SumET',  'Calo_Sig',
        'Track_Final_ET', 'Track_Final_EX', 'Track_Final_EY', 'Track_Final_SumET',
        \
        'Tight_RefJet_ET', 'Tight_RefJet_EX', 'Tight_RefJet_EY', 'Tight_RefJet_SumET',
        'Loose_RefJet_ET', 'Loose_RefJet_EX', 'Loose_RefJet_EY', 'Loose_RefJet_SumET',
        'Tghtr_RefJet_ET', 'Tghtr_RefJet_EX', 'Tghtr_RefJet_EY', 'Tghtr_RefJet_SumET',
        'FJVT_RefJet_ET', 'FJVT_RefJet_EX', 'FJVT_RefJet_EY', 'FJVT_RefJet_SumET',
        \
        'Tight_Muons_ET', 'Tight_Muons_EX', 'Tight_Muons_EY', 'Tight_Muons_SumET',
        'Tight_RefEle_ET', 'Tight_RefEle_EX', 'Tight_RefEle_EY', 'Tight_RefEle_SumET',
        'Tight_RefGamma_ET', 'Tight_RefGamma_EX', 'Tight_RefGamma_EY', 'Tight_RefGamma_SumET',
        \
        'Loose_PVSoftTrk_ET', 'Loose_PVSoftTrk_EX', 'Loose_PVSoftTrk_EY', 'Loose_PVSoftTrk_SumET',
        'Calo_SoftClus_ET', 'Calo_SoftClus_EX', 'Calo_SoftClus_EY', 'Calo_SoftClus_SumET',
        \
        'ActMu', 'NVx_2Tracks', 'NVx_4Tracks', 'PV_NTracks',
        'N_Muons', 'N_Ele', 'N_Gamma', 'N_Jets', 'N_FWD_Jets',
        'SumET_FWD_Jets', 'Sum_JetPU'
    ]

    ## Initialise the model
    model = Model.METNET_Agent(args.name, args.save_dir)

    ## Load up the dataset
    model.setup_dataset(inpt_list, args.data_dir, args.v_frac, args.n_ofiles, args.chnk_size, args.weight_to, args.weight_ratio, args.weight_shift)

    ## Initialise the network
    model.setup_network(args.act, args.depth, args.width, args.nrm, args.drpt)

    ## Setup up the parameters for training
    model.setup_training(args.opt_nm, args.lr, args.reg_loss_nm, args.dst_loss_nm, args.dst_weight, args.grad_clip, args.b_size, args.n_workers)

    ## Run the training loop
    model.run_training_loop()

if __name__ == '__main__':
    main()
