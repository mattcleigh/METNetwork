from Resources import Model
import torch.nn as nn

def main():

    ## Initialise the model
    model = Model.METNET_Agent( name = "METNetTest", save_dir = "Saved_Models" )

    ## Load up the dataset
    model.setup_dataset( data_dir   = "/mnt/scratch/Data/",
                         do_rot     = True,
                         valid_frac = 1e-1,
                         n_ofiles   = 32,  chnk_size = 1024,
                         batch_size = 512, n_workers = 0 )

    ## Initialise the prepost-MLP network
    model.setup_network( act = nn.SiLU(),
                         depth = 5, width = 256, skips = 2,
                         nrm = True, drpt = 0.0 )

    ## Setup up the parameters for training
    model.setup_training( loss_fn = nn.SmoothL1Loss(),
                          lr = 1e-3,
                          clip_grad = 0 )

    ## Load a previous network state
    # model.load( "best" )

    ## Run the training loop
    model.run_training_loop( max_epochs = 1, patience = 5, sv_every = 5 )

    ## Save some prformance metrics using the best version of the network
    model.save_best_perf()

if __name__ == '__main__':
    main()
