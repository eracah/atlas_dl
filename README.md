# atlas_dl
## running the code on Cori

### Training on N examples
* ./run_cori.sh --num_tr N

### Training on all examples
* ./run_cori.sh

#### or

* ./run_cori.sh --num_tr -1

### Testing with the Weights from the Run from 1/19/2017
* ./run_cori.sh --test --load_path ./results/run111/models/net_best_val_loss.pkl

### If you moved the data
* ./run_cori.sh --tr_file path/to/your/tr_file --val_file path/to/your/val_file

#### For test
* ./run_cori.sh --test --test_file path/to/your/test_file --load_path path/to/your/weights

### Running as batch script on Cori
* same commands as above but replace "./run_cori.sh" with "sbatch cori_batch.sl"

### Checking out the Other Command Line Args
* ./run_cori.sh --help

### Running the code on a NERSC notebook:
* module load deeplearning
* go to jupyter.nersc.gov
* open atlas_main.ipynb
* run the cells



## running the code on GPU number K for N example
* ./run_maeve.sh K --num_tr N
