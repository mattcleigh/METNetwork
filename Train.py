import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Model
import torch.nn as nn

def main():

    ## Initialise the model
    model = Model.METNET_Agent( name = "METNet", save_dir = "Saved_Models" )

    ## Load up the dataset
    model.setup_dataset( train_files = "Data/Training/*.h5",
                         test_files = "Data/Training/*.h5",
                         batch_size = 4000, n_workers = 12,
                         stat_file = "Data/Training/stats.csv",
                         x_ids = [4, 9, 14, 19, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 75] )

    ## Initialise the prepost-MLP network
    model.setup_network( act = nn.LeakyReLU(0.2),
                         depth = 5,   width = 128,  skips = 2,
                         nrm = True,  drpt = 0.2 )

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
