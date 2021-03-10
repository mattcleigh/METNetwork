import pandas as pd

file_name = "Data/Training/rot_stats.csv"

df = pd.read_csv( file_name )

for i, c in enumerate(df.columns):
    print(i, c)

print(df.shape)
