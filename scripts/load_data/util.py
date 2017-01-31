
import matplotlib; matplotlib.use("agg")


import h5py
import numpy as np
import os
from os.path import join, exists
from os import makedirs, mkdir



def rename_files(dirpath):
    for fname in os.listdir(dirpath):
        orig_path = join(dirpath, fname)
        suffix = fname.split("test_")[-1]
        new_fpath = join(dirpath, suffix)
        
        print  orig_path, " -> ", new_fpath
        os.rename(orig_path,new_fpath)



def get_atlas_h5group(filepath, key="all_events"):
    h5f = h5py.File(filepath)
    fgroup = h5f[key]
    return fgroup, h5f

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



def preprocess(x, max_abs=None):
    '''a type of sparse preprocessing, which scales everything between -1 and 1 without losing sparsity'''
    #only calculate the statistic using training set
    if max_abs is None:
        max_abs=np.max(np.abs(x))

    #then scale all sets
    x /= max_abs

    return x, max_abs

def preproc_file(fpath, max_val_dict={"weight": None,"hist": None}):
    fgroup, h5f = get_atlas_h5group(fpath)
    hist_normalized, x_max_abs = preprocess(fgroup["hist"][:], max_val_dict["hist"])
    fgroup["hist"][:] = hist_normalized
    nw, w_max_abs = preprocess(fgroup["weight"][:], max_val_dict["weight"])
    
    fgroup.create_dataset(name="normalized_weight", data=nw)
    h5f.close()
    
    return {"weight": w_max_abs,"hist": x_max_abs}





