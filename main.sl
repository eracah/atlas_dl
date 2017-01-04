#!/bin/bash -l

#SBATCH -N 1
#SBATCH -t 30
#SBATCH -p regular
#SBATCH --qos=premium
#SBATCH -C haswell
module load deeplearning

python ./atlas_main.py $@
