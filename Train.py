import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Model
import torch.nn as nn

def main():

    ## Initialise the model
    model = Model.METNET_Agent( name = "METNet", save_dir = "Saved_Models" )


    ## Load up the dataset
    model.setup_dataset( train_files = "Data/Training/rot_sample*.h5",
                         test_files  = "Data/Training/rot_sample*.h5",
                         stat_file   = "Data/Training/rot_stats.csv",
                         n_ofiles    = 1,   chnk_size = 256,
                         batch_size  = 512, n_workers = 8 )

    ## Initialise the prepost-MLP network
    model.setup_network( act = nn.LeakyReLU(0.2),
                         depth = 5,   width = 512,  skips = 0,
                         nrm = False,  drpt = 0.0 )

    ## Setup up the parameters for training
    model.setup_training( loss_fn = nn.SmoothL1Loss(),
                          lr = 1e-4,
                          clip_grad = 0,
                          interactive = True )

    ## Load a previous network state
    # model.load( flag = "" )

    ## Run the training loop
    model.run_training_loop( patience = 20 )

if __name__ == '__main__':
    main()
