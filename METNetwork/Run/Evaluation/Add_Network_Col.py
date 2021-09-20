import glob
import scipy.stats
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

import torch as T
import torch.nn as nn

import METNetwork.Resources.Utils as myUT

def main():

    with T.no_grad():

        ## The main directory to load the data from
        data_folder = '/mnt/scratch/Data/METData/Test/*/'

        ## A list of the network names and the name of the csv file to add to the datafiles
        network_folder = '/mnt/scratch/Saved_Networks/'
        file_suffix = 'models/net_best'
        network_names = [
                          # ('Presentation_Inputs/49506423_0_20_08_21', 'No_Calo'),
                          # ('Presentation_Inputs/49506484_1_20_08_21', 'No_Track'),
                          # ('Presentation_Inputs/49500187_2_20_08_21', 'No_FJVT'),
                          # /
                          # ('Presentation/49484120_0_18_08_21', 'Base'),
                          # ('Presentation/49484124_1_18_08_21', 'Base_Indep'),
                          # ('Presentation/49484127_4_18_08_21', 'Flat'),
                          # ('Presentation/49484133_5_18_08_21', 'Flat_Indep'),
                          # ('Presentation/49496261_4_19_08_21', 'NoRot'),
                          # ('Presentation/49496958_5_19_08_21', 'NoRot_Indep'),
                          # /
                          # ('Presentation_Sampled/49504908_2_20_08_21', 'Dist_Indep'),
                          ('Samples/49525803_0_23_08_21', 'NoRotSinkIndep'),

                          ]

        ## Load the list of input data filesfiles
        all_files = glob.glob(data_folder + '*.train-sample.csv')
        if not all_files:
            raise ValueError('No input files found')

        ## Dont do Zmumu for now
        all_files = [x for x in all_files if 'Zmumu' not in x]

        ## Cycle through the requested networks
        for network_file, network_name in network_names:

            ## Load the network
            net = T.load(Path(network_folder, network_file, file_suffix))

            ## Configure the network for full pass evaluation
            net.to('cuda')
            net.eval()
            net.do_proc = True

            ## Load the input files
            b_size = 10000

            ## Cycle through the input files
            for i, file in enumerate(all_files):
                print('{:.2f}%'.format(100*i/len(all_files)),  file)

                ## Register the buffers
                net_et = []
                net_ex = []
                net_ey = []

                ## Cycle through the batches
                # for batch in pd.read_csv(file, chunksize=b_size, dtype=np.float32):

                try:
                    batch = pd.read_csv(file, dtype=np.float32)
                except:
                    print("FAILED\n")
                    continue

                ## Get the network output for the batch (the final values in the csv are the True_ET, EX, EY and DSID)
                net_inp = T.from_numpy(batch.to_numpy()[:, :-4]).to('cuda')
                net_out = net(net_inp)

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
                output_path = Path(file_path.parent, str(file_path.stem) + '_' + network_name + '.csv')
                net_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
