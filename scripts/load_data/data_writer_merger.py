
import matplotlib; matplotlib.use("agg")


__author__ = 'racah'
import numpy as np
#from sklearn.cross_validation import train_test_split
import sys
import os
from os.path import join, exists
from os import makedirs, mkdir
import h5py
import copy
import pickle



def preprocess(x, max_abs=None):
    '''a type of sparse preprocessing, which scales everything between -1 and 1 without losing sparsity'''
    #only calculate the statistic using training set
    if max_abs is None:
        max_abs=np.max(np.abs(x))

    #then scale all sets
    x /= max_abs

    return x, max_abs



def preproc_file(fpath, max_val_dict={"weight": None}):
    fgroup, h5f = get_atlas_h5group(fpath)
    #fgroup["hist"][:], x_max_abs = preprocess(fgroup["hist"][:], max_val_dict["hist"])
    nw, w_max_abs = preprocess(fgroup["weight"][:], max_val_dict["weight"])
    print nw
    
    fgroup.create_dataset(name="normalized_weight", data=nw)
    h5f.close()
    
    return w_max_abs



def normalize_all_files():
    tr_path = "/global/cscratch1/sd/racah/atlas_h5/train/train.h5"
    val_path = "/global/cscratch1/sd/racah/atlas_h5/train/val.h5"
    test_path = "/global/cscratch1/sd/racah/atlas_h5/test/test.h5"
    mv = {}
    print "tr"
    preproc_file(tr_path)
#     print "val"
#     preproc_file(val_path, mv)
#     print "test"
#     preproc_file(test_path, mv)



normalize_all_files()



def rename_files(dirpath):
    for fname in os.listdir(dirpath):
        orig_path = join(dirpath, fname)
        suffix = fname.split("test_")[-1]
        new_fpath = join(dirpath, suffix)
        
        print  orig_path, " -> ", new_fpath
        os.rename(orig_path,new_fpath)
        



#rename_files("/global/cscratch1/sd/racah/atlas_h5/test")



def get_atlas_h5group(filepath, key="all_events"):
    h5f = h5py.File(filepath)
    fgroup = h5f[key]
    return fgroup, h5f
    



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
    print base
    print "making new file..."
    make_new_file(base, new_fpath)
    return base
        
    



def make_empty_dict_of_file(filepath):
    fgroup, h5f = get_atlas_h5group(filepath)
    ed = {k : np.empty(tuple([0] + list(v.shape[1:]))) for k,v in fgroup.iteritems()}
    ed["y"] = np.empty((0,))
    h5f.close()
    return ed
    



def concat_two_dicts(base,addition):
    for k,v in addition.iteritems():
        if len(v.shape) == 1:
            base[k] = np.hstack((base[k], addition[k]))
        else:
            base[k] = np.vstack((base[k], addition[k]))
    return base



def get_data_dict_from_h5group(h5group, sig=False):
    d = {}
    for k,v in h5group.iteritems():
        d[k] = v[:]
    num_events = d[d.keys()[0]].shape[0]
    d["y"] = np.zeros((num_events,)) if not sig else np.ones((num_events,))
    return d



def make_new_file(dic, new_fpath):
    newf = h5py.File(new_fpath)
    newg = newf.create_group("all_events")
    for k,v in dic.iteritems():
        newg[k] = v
    newf.close()



def split_train_test_files(file_path_list, test_prop=0.2):
    
    def add_to_file(file_name, data_dict):
        f = h5py.File(file_name, "w")
        group = f.create_group("all_events")
        for k in data_dict:
            group[k] = data_dict[k]
        f.close()
        
    for file_path in file_path_list:
        print file_path
        h5f = h5py.File(file_path)
        all_events = h5f["all_events"]
        num_events = all_events["hist"].shape[0]
        
        num_test = int(test_prop * num_events)
        
        test_file_name = join(os.path.dirname(file_path),"val_" + os.path.basename(file_path))
        train_file_name = join(os.path.dirname(file_path),"train_" + os.path.basename(file_path))
        
        inds = np.arange(num_events)
        np.random.RandomState(11).shuffle(inds)
        raw_data = {k:all_events[k][:] for k in all_events.keys()}
        te_data = {k:raw_data[k][inds[:num_test]] for k in all_events.keys()}
        tr_data = {k:raw_data[k][inds[num_test:]] for k in all_events.keys()}
        add_to_file(test_file_name, te_data)
        add_to_file(train_file_name, tr_data)
        
        
    
        
    
        

def run_split():
    split_train_test_files(file_path_list=["/global/cscratch1/sd/racah/atlas_h5/train/train.h5"])
    





