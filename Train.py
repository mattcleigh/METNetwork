from Resources import Model
import torch.nn as nn

def main():

    ## Initialise the model
    model = Model.METNET_Agent( name = "METNet", save_dir = "Saved_Models" )

    ## Load up the dataset
    model.setup_dataset( data_dir   = "../Data/Raw/",
                         stat_file  = "../Data/Raw/stats.csv",
                         valid_frac = 0.1,
                         n_ofiles   = 32,  chnk_size = 1024,
                         batch_size = 512, n_workers = 12 )

    ## Initialise the prepost-MLP network
    model.setup_network( n_in = 76,
                         act = nn.SiLU(),
                         depth = 5, width = 256, skips = 0,
                         nrm = False, drpt = 0.0 )

    ## Setup up the parameters for training
    model.setup_training( loss_fn = nn.SmoothL1Loss(),
                          lr = 1e-5,
                          clip_grad = 0 )

    ## Load a previous network state
    # model.load( "best" )

    ## Run the training loop
    model.run_training_loop( patience = 10, sv_every = 5 )

    ## Save some prformance metrics using the best version of the network
    model.save_perf("best")

if __name__ == '__main__':
    main()
