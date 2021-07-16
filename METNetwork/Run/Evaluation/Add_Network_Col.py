import glob
import scipy.stats
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

import torch as T
import torch.nn as nn

from METNetwork.Resources import Model
import METNetwork.Resources.Utils as myUT

def main():

    with T.no_grad():

        data_folder = '/home/matthew/Documents/PhD/Data/METData/Test/*ZZ*/'

        network_names = [ ('Test', '/home/matthew/Documents/PhD/Saved_Networks/tmp/IndepDrp/') ]

        ## Load the list of files
        all_files = glob.glob( data_folder + '*sample.csv' )
        if not all_files:
            print('No input files found')
            exit()

        ## Cycle through the requested networks
        for network_name, network_file in network_names:

            ## Load the network
            model = Model.METNET_Agent('IndepDrp', '/home/matthew/Documents/PhD/Saved_Networks/tmp/')
            model.load('best')

            ## Configure the network for full pass evaluation
            model.net.to('cuda')
            model.net.eval()
            model.net.do_proc = True

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

                    ## Get the network output for the batch (the final values in the csv are the True_ET, EX, EY and DSID)
                    net_inp = T.from_numpy(batch.to_numpy()[:, :-4]).to('cuda')
                    net_out = model.net(net_inp)

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
