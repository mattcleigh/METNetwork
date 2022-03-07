"""
Splits the data in ttbar allhad using a weight function
"""


import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from mattstools.utils import undo_mid

def main():

    ## The program settings
    inpt_dir = "/mnt/scratch/Data/METData/all_had/user.mleigh.23_02_22.WithSigAndEtaFix.ttbar_410471_EXT0/"
    out_passed = "/mnt/scratch/Data/METData/all_had/Train/"
    out_failed = "/mnt/scratch/Data/METData/all_had/Test/"
    spline_file = "/home/users/l/leighm/METNetwork/METNetwork/output/input_histograms/allhad_spline.csv"

    ## Load the spline function and get the edges of the bins
    spline = pd.read_csv(spline_file)
    bin_edges = undo_mid(spline.bins.to_numpy())
    spline = spline["allhad sample weights"].to_numpy()

    ## Get the list of csv files in the input folder
    input_files = list(Path(inpt_dir).glob("*"))

    ## Cycle through the input files in the list
    for file in tqdm(input_files):

        ## Load using pandas
        df = pd.read_csv(file)

        ## Split into bins
        bins = pd.cut(df.True_ET/1000, bin_edges, labels=False, include_lowest=True)

        ## Use the bins to calculate the spline values
        sp_vals = spline[bins.to_numpy(dtype=np.int64)]

        ## Use a random test to see if passed
        rndm = np.random.rand(len(sp_vals))
        is_passed = rndm < sp_vals

        ## Save two versions of the dataframe
        df[is_passed].to_csv(Path(out_passed, Path(inpt_dir).name , file.name), index=False)
        df[~is_passed].to_csv(Path(out_failed, Path(inpt_dir).name , file.name), index=False)

if __name__ == "__main__":
    main()
