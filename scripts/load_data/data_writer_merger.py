
import matplotlib; matplotlib.use("agg")


__author__ = 'racah'
import numpy as np
import sys
import os
from os.path import join, exists
from os import makedirs, mkdir
import h5py
import copy
import pickle
from util import get_atlas_h5group, make_empty_dict_of_file, concat_two_dicts,                     get_data_dict_from_h5group, make_new_file,                     preprocess, preproc_file
        
import argparse



def merge_files(dirpath, new_fpath):
    files = [join(dirpath,f) for f in os.listdir(dirpath)]
    base = make_empty_dict_of_file(files[0])
    for fpath in files:
        if "RPV" in fpath:
            sig=True
        else:
            sig=False
        print fpath
        fgroup, h5f = get_atlas_h5group(fpath)
        
        d = get_data_dict_from_h5group(fgroup, sig)
        base = concat_two_dicts(base,d)
    
        h5f.close()
    print "making new file..."
    make_new_file(base, new_fpath)
    #return base
        
    



def split_train_test_files(file_path,test_name="test", val_name="val", train_name="train",suffix="", test_prop=0.2):
    
    def add_to_file(file_name, data_dict):
        #overwrites file?
        f = h5py.File(file_name, "w")
        group = f.create_group("all_events")
        for k in data_dict:
            group[k] = data_dict[k]
        f.close()
        

    print file_path
    h5f = h5py.File(file_path)
    all_events = h5f["all_events"]
    num_events = all_events["hist"].shape[0]

    num_test = int(test_prop * num_events)
    num_val = num_test
    test_file_name = join(os.path.dirname(file_path),test_name + suffix + ".h5")
    train_file_name = join(os.path.dirname(file_path),train_name + suffix+ ".h5")
    val_file_name = join(os.path.dirname(file_path),val_name + suffix+ ".h5")
    
    
    inds = np.arange(num_events)
    np.random.RandomState(11).shuffle(inds)
    raw_data = {k:all_events[k][:] for k in all_events.keys()}
    te_data = {k:raw_data[k][inds[:num_test]] for k in all_events.keys()}
    val_data = {k:raw_data[k][inds[num_test:2*num_test]] for k in all_events.keys()}
    tr_data = {k:raw_data[k][inds[2*num_test:]] for k in all_events.keys()}
    add_to_file(test_file_name, te_data)
    add_to_file(train_file_name, tr_data)
    add_to_file(val_file_name, val_data)
    return {"tr": train_file_name, "te": test_file_name, "val": val_file_name}
        
    



def normalize_all_files(tr_path, val_path, test_path):
    print "tr"
    tr_stats = preproc_file(tr_path)
    print "val"
    _ = preproc_file(val_path, tr_stats)
    print "test"
    _ = preproc_file(test_path, tr_stats)



def parse_args():
    if len(sys.argv) > 2:
        assert "jupyter" not in sys.argv[2], "can't run this from a notebook!"
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_path",type=str, help="path where the initial files are" )
    ap.add_argument("--dest_path", type=str, help="path where you want tr, val, test files")
    ap.add_argument("--suffix", default="", type=str, help="suffix for describing tr val test files")
    args = ap.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    source_dir = args.source_path
    dest_dir = args.dest_path
    suffix = args.suffix
    
    # merge all data in source_dir directory into a file in the dest_dir diretory
    # called "all_data_merged<your_suffix>.h5"
    merged_fpath = join(dest_dir, "all_data_merged" + suffix + ".h5" )
    merge_files(source_dir, merged_fpath)
    
    #split the merged file into train and test and val (60-20-20 split)
    tr_te_val = split_train_test_files(merged_fpath,suffix=suffix)
    
    
    #now normalize val and test based off of train statistics
    normalize_all_files(tr_path=tr_te_val["tr"], test_path=tr_te_val["te"], val_path=tr_te_val["val"])

