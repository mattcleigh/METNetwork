from Resources import Model
import torch.nn as nn

def main():

    ## Initialise the model
    model = Model.METNET_Agent( name = "test_network", save_dir = "Saved_Models" )

    ## Load up the dataset
    model.setup_dataset( data_dir   = "../Data/",
                         do_rot     = True,
                         do_weights = False,
                         valid_frac = 1e-2,
                         n_ofiles   = 2,   chnk_size = 1024,
                         batch_size = 512, n_workers = 4 )

    ## Initialise the prepost-MLP network
    model.setup_network( act = nn.LeakyReLU(0.1),
                         depth = 2, width = 256, skips = 1,
                         nrm = False, drpt = 0.0 )

    ## Setup up the parameters for training
    model.setup_training( loss_fn = nn.SmoothL1Loss(),
                          lr = 1e-3,
                          clip_grad = 0 )

    ## Load a previous network state
    # model.load( "best" )

    ## Run the training loop
    model.run_training_loop( max_epochs = 50, patience = 5, sv_every = 5 )

if __name__ == '__main__':
    main()
