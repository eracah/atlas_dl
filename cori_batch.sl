#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -C haswell
#SBATCH -p regular

./run_cori.sh $@
