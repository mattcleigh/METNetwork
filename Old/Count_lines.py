import glob

folder = "../Data/Grid/"
file_list = glob.glob( folder + "/*.csv" )

total = 0
for file in file_list:
    with open(file) as f:
        total += sum(1 for line in f) - 1
    print(total)

Grid_Train.py --name GridTest --do_rot False --bsize 256 --depth 2 --width 24 --skips 1 --nrm False --lr 1e-3
