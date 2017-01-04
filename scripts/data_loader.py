
import matplotlib; matplotlib.use("agg")


__author__ = 'racah'
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import sys
import os
from os.path import join, exists
from os import makedirs, mkdir
import time
import h5py
#from helper_fxns import suppress_stdout_stderr
import copy
import pickle
import re
#sys.path.append('/global/homes/w/wbhimji/cori-envs/nersc-rootpy/lib/python2.7/site-packages/')



def shuffle(kwargs):
    inds = np.arange(kwargs[kwargs.keys()[0]].shape[0])

    #shuffle data
    rng = np.random.RandomState(7)
    rng.shuffle(inds)
    return {k:v[inds] for k,v in kwargs.iteritems()}

def split_train_val(prop, kwargs):
    tr_prop = prop
    inds = np.arange(kwargs[kwargs.keys()[0]].shape[0])
    #split train, val, test
    num_tr_ex = int((tr_prop*len(inds)))
    dind = {}
    dind["tr"] = inds[:num_tr_ex]
    #dind["test"] = inds[num_tr_ex:num_tr_ex + num_val_ex]
    
    dind["val"] = inds[num_tr_ex:]
    
    data = {}
    for typ, inds in dind.iteritems():
        data[typ] = {k:v[inds] for k,v in kwargs.iteritems()}
        
        
    #dict of dicts, where key is tr, val or test and value is dict of x,y,w,psr, etc.
    return data



def preprocess(x, max_abs=None):
    '''a type of sparse preprocessing, which scales everything between -1 and 1 without losing sparsity'''
    #only calculate the statistic using training set
    if max_abs is None:
        max_abs=np.max(np.abs(x))

    #then scale all sets
    x /= max_abs

    return x, max_abs



class DataLoader(object):
    def __init__(self, 
                 bg_cfg_file = './config/BgFileListAug16.txt',
                 sig_cfg_file='./config/SignalFileListAug16.txt',
                 events_fraction=0.1,
                 preproc=True,
                 tr_prop=0.8,
                 seed=3, 
                 test = False):
        
        
        self.bg_files = bg_cfg_file if isinstance(bg_cfg_file, list) else [bg_cfg_file]
        self.sig_files = sig_cfg_file if isinstance(sig_cfg_file, list) else [sig_cfg_file]
        self.all_files = self.bg_files + self.sig_files
        self.test = test
        self.preproc = preproc
        self.tr_prop = tr_prop
        self.seed = seed
        assert events_fraction >= 0 and events_fraction <= 1. , "whoa events between 0 and 1!"
        self.events_fraction = events_fraction
        
         
          
    def vstack_all(self, data):
        for k,v in data.iteritems():
            data[k] = np.vstack(tuple(v))
        return data
    
    
 
        
            
    def make_hist(self, d):
        
        return np.histogram2d(d['clusphi'],d['cluseta'], bins=(self.phi_bins, self.eta_bins),
                              weights=d["cluse"], range=[self.phi_range,self.eta_range])[0] 

   

        
    def _grab_hdf5_events(self,file_, start):
        h5f = h5py.File(file_)
        print file_
        
        all_events = h5f["all_events"]
        num_events_in_file = all_events["hist"].shape[0]
        
        num_events = int(np.ceil(self.events_fraction * num_events_in_file))
            
        arr_slice = slice(0,num_events)
        x,w,psr = [np.expand_dims(all_events[k][arr_slice],axis=1) for k in ["hist", "weight", "passSR"]]
        filename = os.path.basename(file_)
        jz = [int(filename.split("JZ")[-1].split(".h5")[0]) if "jet" in filename else -1]
        rpv1 = [int(filename.split(".h5")[0].split("_")[3]) if "jet" not in filename else -1]
        rpv2 = [int(filename.split(".h5")[0].split("_")[4]) if "jet" not in filename else -1]
        lbls = dict(jz=jz, rpv1=rpv1, rpv2=rpv2)
        lbls = {k:np.expand_dims(num_events*v, axis=1) for k,v in lbls.iteritems()}
        data =  dict(x=x, w=w, psr=psr)  
        data.update(lbls)
        return data
        
    def grab_file(self, file_,files_dict):
        file_dict = self._grab_hdf5_events(file_, start=0)
        
        for k, v in file_dict.iteritems():
            files_dict[k].append(v)
        
        
        return files_dict
    
    
    def grab_events(self, file_list, start=0):
        files_dict = {k:[] for k in ["w", "x", "psr", "jz", "rpv1", "rpv2"]}
        
        if len(file_list) > 0:

            for file_ in file_list:
                files_dict = self.grab_file(file_, files_dict)

            files_dict = self.vstack_all(files_dict)

            
            
        return files_dict
    
    def get_data_block(self):
        bg = self.grab_events(self.bg_files)
        sig = self.grab_events(self.sig_files)
        if len(sig[sig.keys()[0]]) > 0:
            if len(bg[bg.keys()[0]]) > 0:
                data = {k:np.vstack((bg[k], sig[k])) for k in bg.keys()}
            else:
                data = sig
        elif len(bg[bg.keys()[0]]) > 0:
            data = bg
        else:
            assert False, "you got no data"
        

        
        num_data_bg = len(bg["w"])
        num_data_sig = len(sig["w"])

        
        # 1 means signal, 0 means background
        data["y"] = np.zeros((num_data_bg + num_data_sig)).astype('int32')

        
        #make the last half signal label
        data["y"][num_data_bg:] = 1
   
        return data
        

    def load_data(self):
        t = time.time()
        data = self.get_data_block()
        print time.time() - t
        data = shuffle(data)
        if not self.test:
            data = split_train_val(self.tr_prop, data)
        if self.preproc:
            data = self.preprocess(data, keys=["x", "w"], raw=["w"])

               
        return data

    
    def preprocess(self, data, keys=["x"], raw=["w"]):
        
        if not self.test:
            for k in keys:
                tr = copy.deepcopy(data["tr"][k])
                val = copy.deepcopy(data["val"][k])
                if k in raw:
                    data["tr"]["raw_"+ k] = tr
                    data["val"]["raw_" + k] = val
                data["tr"][k],tm = preprocess(data["tr"][k])
                data["val"][k], _ = preprocess(data["val"][k],tm)

        else:
            for k in keys:
                dat = copy.deepcopy(data[k])
                if k in raw:
                    data["raw_"+ k] = dat
                data[k], _  = preprocess(data[k]) 
            data = {"test":data}
        return data
        
    def iterate_data(self, batch_size=128):
#         if self.num_each < batch_size / 2:
#             batch_size = 2 * self.num_each
#         #only support for hdf5
#         for i in range(0, self.num_each, batch_size / 2):
#             x_bg = self.grab_events(self.bg_files, batch_size / 2, i)
#             x_sig = self.grab_events(self.sig_files, batch_size / 2, i)
        pass



#does same thing as DataLoader, but makes sure train set is all bg
class AnomalyLoader(DataLoader):
    def __init__(self, **kwargs):
        DataLoader.__init__(self, **kwargs)
    
    
    def load_data(self):
        data = self.get_data_block()
        data = shuffle(data)
        if not self.test:
            data = self.split_train_val_anom(self.tr_prop, data)
        data = self.preprocess(data)
        return data

                    
    def split_train_val_anom(self,prop, data):
        tr_prop = prop
        
        all_inds = np.arange(data[data.keys()[0]].shape[0])
        bg_inds = all_inds[data["y"] == 0]
        sig_inds = all_inds[data["y"] == 1]
        
        #split train, val
        #only use background for train
        num_tr_ex = int((tr_prop*len(bg_inds)))
        
        dind = {}
        dind["tr"] = bg_inds[:num_tr_ex]


        dind["val"] = np.concatenate((bg_inds[num_tr_ex:], sig_inds))

        final_data = {}
        for typ, inds in dind.iteritems():
            final_data[typ] = {k:v[inds] for k,v in data.iteritems()}


        #dict of dicts, where key is tr, val or test and value is dict of x,y,w,psr, etc.
        return final_data



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
        
        test_file_name = join(os.path.dirname(file_path),"test_" + os.path.basename(file_path))
        train_file_name = join(os.path.dirname(file_path),"train_" + os.path.basename(file_path))
        
        inds = np.arange(num_events)
        np.random.RandomState(11).shuffle(inds)
        raw_data = {k:all_events[k][:] for k in all_events.keys()}
        te_data = {k:raw_data[k][inds[:num_test]] for k in all_events.keys()}
        tr_data = {k:raw_data[k][inds[num_test:]] for k in all_events.keys()}
        add_to_file(test_file_name, te_data)
        add_to_file(train_file_name, tr_data)
        
        
    
        
    
        



def run_split():
    h5_prefix = "/global/cscratch1/sd/racah/atlas_h5"
    bg_cfg_file=[join(h5_prefix, "jetjet_JZ%i.h5"% (i)) for i in range(3,12)]
    sig_cfg_file=[join(h5_prefix, "GG_RPV10_1400_850.h5")]
    file_list = bg_cfg_file + sig_cfg_file
    print file_list
    #split_train_test_files(file_path_list=file_list)
    



if __name__=="__main__":
    #run_split()
    h5_prefix = "/global/cscratch1/sd/racah/atlas_h5"
    

    dl = DataLoader(bg_cfg_file=[join(h5_prefix, "train_jetjet_JZ%i.h5"% (i)) for i in range(4,12)],
                    sig_cfg_file=join(h5_prefix, "train_GG_RPV10_1400_850.h5"),
               events_fraction=0.05, test=False, preproc=True)

    data= dl.load_data()






