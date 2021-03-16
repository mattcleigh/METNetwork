#!/bin/bash

#SBATCH --job-name=prepare-dataset-job-submit
#SBATCH	--partition=shared-cpu,private-dpnc-cpu
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

cd /home/users/l/leighm/METNetwork

module load GCCcore/8.2.0 Singularity/3.4.0-Go-1.12

srun singularity exec -B /home/users/l/leighm/scratch:/mnt/scratch /home/users/l/leighm/Images/met-network-rep_latest.sif python Prepare_Trainset.py \
        --input_dir  /mnt/scratch/Data/user.mleigh.09_03_21.FINAL.ttbar_410470_EXT0/ \
        --output_dir /mnt/scratch/Data/Rotated/ \
        --do_rot     True \

srun singularity exec -B /home/users/l/leighm/scratch:/mnt/scratch /home/users/l/leighm/Images/met-network-rep_latest.sif python Prepare_Trainset.py \
        --input_dir  /mnt/scratch/Data/user.mleigh.09_03_21.FINAL.ttbar_410470_EXT0/ \
        --output_dir /mnt/scratch/Data/Raw/ \
        --do_rot     False \
