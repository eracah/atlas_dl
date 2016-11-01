# atlas_dl

## Running the code on a notebook:

* mkdir ~/.ipython/kernels/deeplearning
* cp kernel.json ~/.ipython/kernels/deeplearning
* go to ipython.nersc.gov and change kernel to deeplearning
* run atlas_main.ipynb


## Running code from shell or bash script:
* module load deeplearning
* python atlas_main.py -e \<num_epochs\> -n \<num_events\>
