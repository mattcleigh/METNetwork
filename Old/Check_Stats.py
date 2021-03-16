import pandas as pd

df = pd.read_hdf("/home/matthew/Documents/PhD/Data/Rotated/rot_sample-00.h5")
print(len(df))
print(df.mean(axis=0))
print(df.std(axis=0))
