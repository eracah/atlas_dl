#!/bin/bash

module load python/3.5-anaconda
source activate deeplearning
python atlas_main_caffe.py $@
