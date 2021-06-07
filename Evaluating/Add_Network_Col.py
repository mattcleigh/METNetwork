import sys
home_env = '../'
sys.path.append(home_env)

import glob
import scipy.stats
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

import torch as T
import torch.nn as nn

from Resources import Model
from Resources import Utils as myUT
from Export import Enveloped_Model

def main():

    with T.no_grad():

        network_name = "FlatSinkhorn"

        ## Load the network
        model = Model.METNET_Agent( name = "METNet_PLSWRK_LY_23_05_21_46858343_0",
                                    save_dir = "../../Saved_Networks/PLSWRK/")
        model.setup_network( act = nn.SiLU(),
                             depth = 5, width = 1024, skips = 0,
                             nrm = True, drpt = 0.0, dev = "cuda" )
        model.load("best", get_opt = False)
        model.network.eval()

        ## Load the stat file for pre-post processing
        stats = np.loadtxt( model.save_dir+model.name+"/stat.csv",
                            skiprows=1,
                            delimiter=",",
                            dtype=np.float32 )
        stats = T.from_numpy(stats).to('cuda')

        ## Envelop the model with its pre and post processing steps
        env_model = Enveloped_Model(model.network, stats)

        ## Load the input files
        data_folder = "../../Data/METData/Raw/ttbar/"
        all_files = glob.glob( data_folder + '*sample.csv' )
        b_size = 4096

        ## Cycle through the input files
        for file in all_files:

            ## Register the buffers
            net_et = []
            net_ex = []
            net_ey = []

            ## Cycle through the batches
            for batch in tqdm(pd.read_csv(file, chunksize=b_size, dtype=np.float32)):

                ## Get the network output for the batch
                net_inp = T.from_numpy(batch.to_numpy()[:, :-4]).to("cuda")
                net_out = env_model( net_inp )

                ## Fill in the buffers
                net_et.append( T.linalg.norm(net_out, axis=1).cpu() )
                net_ex.append( net_out[:, 0].cpu() )
                net_ey.append( net_out[:, 1].cpu() )

            ## Combine the buffers into numpy arrays
            net_et = np.concatenate( net_et, axis = 0 )
            net_ex = np.concatenate( net_ex, axis = 0 )
            net_ey = np.concatenate( net_ey, axis = 0 )

            net_df = pd.DataFrame( { network_name+'_ET': net_et,
                                     network_name+'_EX': net_ex,
                                     network_name+'_EY': net_ey } )

            ## Save the dataframe
            file_path = Path(file)
            output_path = Path( file_path.parent, str(file_path.stem) + '_' + network_name + '.csv')
            net_df.to_csv( output_path, index=False )




if __name__ == '__main__':
    main()
