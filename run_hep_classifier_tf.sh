#!/bin/bash
#SBATCH -p regular
#SBATCH -A nstaff
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 6:00:00
#SBATCH -J hep_train_tf

#load python
#module load python
#source activate thorstendl

#OpenMP stuff
export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

python ./scripts/hep_classifier_tf.py