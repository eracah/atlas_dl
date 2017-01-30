# atlas_dl
## Running the code on Cori

### Training on N examples
~~~~
./run_cori.sh --num_tr N
~~~~

### Training on all examples
~~~~
./run_cori.sh
~~~~
#### or
~~~~
./run_cori.sh --num_tr -1
~~~~
### Testing with the Weights from the Run from 1/19/2017
~~~~
./run_cori.sh --test --load_path ./results/run111/models/net_best_val_loss.pkl
~~~~
### If you moved the data
~~~~
./run_cori.sh --tr_file path/to/your/tr_file --val_file path/to/your/val_file
~~~~

#### For test
~~~~
./run_cori.sh --test --test_file path/to/your/test_file --load_path path/to/your/weights
~~~~

### Running as batch script on Cori

* same commands as above but replace "./run_cori.sh" with "sbatch cori_batch.sl"

### Checking out the Other Command Line Args
~~~~
./run_cori.sh --help
~~~~

### Running the code on a NERSC notebook:
~~~~
module load deeplearning
~~~~
* go to jupyter.nersc.gov
* open atlas_main.ipynb
* run the cells

### Merging all events, then splitting them into train, val and test
~~~~
./preproc_files.sh --source_path path/where/initial/input/files/are --dest_path path/where/you/want/to/put/trvaltest/files --suffix "string_to_append_to_files_for_your_own_benefit"
~~~~
* if you pick suffix to be "_run5", then in your dest_path will be four files:
   * all_data_merged_run5.h5
   * train_run5.h5
   * test_run5.h5
   * val_run5.h5
   
* you can delete all_data_merged_run5.h5, which is all your data in one file, to save space or you can keep it for maybe a new tr,val,test split later
   

## running the code on GPU number K for N examples
~~~~
./run_maeve.sh K --num_tr N
~~~~
