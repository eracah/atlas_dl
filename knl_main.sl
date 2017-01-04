#!/bin/bash -l

#SBATCH -N 1
#SBATCH -t 30
#SBATCH -C knl
module load deeplearning

python ./atlas_main.py $@
