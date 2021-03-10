import os
import numpy as np
import pandas as pd

import torch as T

def main():

    folder = "Data/Training/"
    inpt_file = "Training_Dataset.csv"
    otpt_file = "stats.csv"

    ## Getting the data tensor
    df = pd.read_csv( os.path.join( folder, inpt_file ), nrows=10 )
    data = df.values.astype(np.float32)
    data = T.as_tensor( data, dtype = T.float32 )

    ## The indicies of the x, y components and the et (all next to each other)
    x_idx = []
    for i, col in enumerate(df.columns):
        print(i, col)
        if "EX" in col:
            x_idx.append(i)
    print(x_idx)
    exit()

    x_idx = np.array([4, 9, 14, 19, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 75])
    y_idx = x_idx + 1

    ## Loading the input vector and the angle (clone to not modify original data)
    input_vec = data[0:3, :].clone()
    angle = data[0:3, 0:1].clone()
    print( "Input shape (btc, vec): ", input_vec.shape )
    print( "Angle shape (btc, vec): ", angle.shape )

    ## Saving the magnitudes (for verifying that the rotation worked)
    cal_mags = T.sqrt( input_vec[:, x_idx]**2 + input_vec[:, y_idx]**2 )

    ## Perform the rotation (need temp tensors so I dont use rotated x vals, neater to use both)
    rotated_x           =   input_vec[:, x_idx] * T.cos(angle) + input_vec[:, y_idx] * T.sin(angle)
    input_vec[:, y_idx] = - input_vec[:, x_idx] * T.sin(angle) + input_vec[:, y_idx] * T.cos(angle)
    input_vec[:, x_idx] = rotated_x

    ## Calculate the new magnitude and check if it is close to origonal
    rot_mags = T.sqrt( input_vec[:, x_idx]**2 + input_vec[:, y_idx]**2 )
    print( "Max difference to between raw and rot:", T.max(cal_mags-rot_mags).item() )

    ## Show which indicies changed (to verify only the correct components were changed)
    nonz = T.nonzero( input_vec - data[0:3, :] )
    for nz in nonz:
        idx = nz[1].item()
        flag = "BOO"
        if   idx in x_idx: flag = "YAY"
        elif idx in y_idx: flag = "YAY"
        print( idx, flag )

if __name__ == '__main__':
    main()
