import glob
import scipy.stats
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

import torch as T
import torch.nn as nn

import METNetwork.Resources.Model as myMD
import METNetwork.Resources.Utils as myUT
from Export import Enveloped_Model

def main():

    with T.no_grad():

        data_folder = '/mnt/scratch/Data/METData/Raw/user.mleigh.02*/'

        network_names = [ ('Normal', 'METNet_CutStudy_20_06_21_47688528_9', False, False),
                          ('NoCalo', 'METNet_CutStudy_20_06_21_47688250_6', True, False),
                          ('NoTrack', 'METNet_CutStudy_20_06_21_47688245_3', False, True),
                          ('Neither', 'METNet_CutStudy_20_06_21_47688241_0', True, True) ]

        ## Load the list of files
        all_files = glob.glob( data_folder + '*sample.csv' )
        if not all_files:
            print('No input files found')
            exit()

        ## Cycle through the requested networks
        for network_name, network_file, cc, ct in network_names:

            ## Load the network
            model = myMD.METNET_Agent( name = 'METNet_CutStudy_20_06_21_47688528_9',
                                       save_dir = '/mnt/scratch/Saved_Networks/CutStudy/')
            model.setup_network( cc, ct,
                                 act = nn.SiLU(),
                                 depth = 5, width = 1024, skips = 0,
                                 nrm = True, drpt = 0.0, dev = 'cuda' )
            model.load('best', get_opt = False)
            model.network.eval()

            ## Load the stat file for pre-post processing
            stats = np.loadtxt( model.save_dir+model.name+'/stat.csv',
                                skiprows=1,
                                delimiter=',',
                                dtype=np.float32 )
            stats = T.from_numpy(stats).to('cuda')

            ## Envelop the model with its pre and post processing steps
            env_model = Enveloped_Model(model.network, stats)

            ## Load the input files
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
                    net_inp = T.from_numpy(batch.to_numpy()[:, :-4]).to('cuda')
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
