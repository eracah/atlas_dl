#!/bin/bash
#SBATCH -p regular
#SBATCH -A nstaff
#SBATCH -N 5
#SBATCH -C haswell
#SBATCH -t 8:00:00
#SBATCH -J hep_train_tf

#OpenMP stuff
#export OMP_NUM_THREADS=32
#export OMP_NUM_THREADS=66
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

srun -n 5 -c 64 -u python hep_classifier_tf_train.py
