
import matplotlib; matplotlib.use("agg")


__author__ = 'tkurth'
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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



class DataIterator(object):
    def __init__(self, filelist, batch_size=128, shuffle=True, keys={"datakey": "data", "labelkey": "label", "normweightkey":"normweight", "weightkey":"weight"}):
        #keys
        self.keys=keys
        #batchsize and indices
        self.batch_size=batch_size
        #store the filelist
        self.files=filelist
        self.num_files=len(self.files)
        #store the shuffle state
        self.shuffle=shuffle
        #file and event indices:
        self.file_index=0
        self.event_index=0
        #hgroup:
        self.hgroup={}
        
        #determine how many events we have:
        self.num_events=0
        for fname in self.files:
            f=h5py.File(fname,'r')
            count=f[self.keys['labelkey']].shape[0]
            f.close()
            self.num_events+=count
        
        #shuffle files
        if self.shuffle:
            np.random.shuffle(self.files)
            
        #load the initial bunch of data
        self.load_next_file()
    
    
    #iterator
    def __iter__(self):
        return self
    
    
    #load next file logic
    def load_next_file(self):
        #open file
        f=h5py.File(self.files[self.file_index],'r')
        #load data from file
        self.data=f[self.keys['datakey']].value
        self.label=f[self.keys['labelkey']].value
        self.nweight=f[self.keys['normweightkey']].value
        self.weight=f[self.keys['weightkey']].value
        #close file
        f.close()
        
        #datalength:
        self.dlength=self.data.shape[0]
        
        #shuffle data if requested
        if self.shuffle:
            reindex=np.random.permutation(self.dlength)
            self.data=self.data[reindex,:,:,:]
            self.label=self.label[reindex]
            self.nweight=self.nweight[reindex]
            self.weight=self.weight[reindex]
    
    
    #next function
    def __next__(self):
        #grep data
        #upper index
        upper=np.min([self.dlength,self.event_index+self.batch_size])
        #load data
        tmpdata=self.data[self.event_index:upper,:,:,:].astype("float32")
        tmplabel=self.label[self.event_index:upper].astype("int32")
        tmpweight=self.weight[self.event_index:upper].astype("float32")
        tmpnweight=self.nweight[self.event_index:upper].astype("float32")
        #load new file if needed:
        if self.dlength<=(self.event_index+self.batch_size):
            self.file_index+=1
            
            #check if the epoch is over
            if self.file_index>=self.num_files:
                #shuffle if requested
                if self.shuffle:
                    np.random.shuffle(self.files)
                #reset indices
                self.event_index=0
                self.file_index=0
                #prefetch next
                self.load_next_file()
                #stop the iteration here
                raise StopIteration
            else:
                #prefetch the file
                self.load_next_file()
                #fetch the missing data:
                rlength=self.batch_size-tmpdata.shape[0]
                tmpdata=np.concatenate([tmpdata,self.data[0:rlength,:,:,:]],axis=0)
                tmplabel=np.concatenate([tmplabel,self.label[0:rlength]],axis=0)
                tmpweight=np.concatenate([tmpweight,self.weight[0:rlength]],axis=0)
                tmpnweight=np.concatenate([tmpnweight,self.nweight[0:rlength]],axis=0)
                self.event_index=rlength
        else:
            self.event_index+=self.batch_size
            
        #return result
        self.hgroup={'hist': tmpdata, 'y': tmplabel, 'weight': tmpweight, 'norm_weight': tmpnweight}
        return self.hgroup
        
        
    #def iterate(self):
    #    
    #    #shuffle files
    #    if self.shuffle:
    #        self.files=np.random.shuffle(self.files)
    #    #set step-size to batch-size
    #    step_size=self.batch_size
    #    
    #    #iterate over files
    #    for fname in self.files:
    #        
    #        #open file
    #        f=h5py.File(fname,'r')
    #        #load data from file
    #        data=np.random.shuffle(f[keys['datakey']],axis=0).values
    #        label=np.random.shuffle(f[keys['labelkey']],axis=0).values
    #        nweight=np.random.shuffle(f[keys['normweightkey']],axis=0).values
    #        weight=np.random.shuffle(f[keys['weightkey']],axis=0).values
    #        #close file
    #        f.close()
    #        #datalength:
    #        datalength=data.shape[0]
    #        
    #        #shuffle data if requested
    #        if self.shuffle:
    #            reindex=np.random.permutation(datalength)
    #            data=data[reindex,:,:,:]
    #            label=label[reindex]
    #            nweight=nweight[reindex]
    #            weight=weight[reindex]
    #        
    #        #iterate over entries
    #        for event_index in range(0,datalength,step_size):
    #            #upper index
    #            upper=np.max([datalength,event_index+step_size])
    #            #load data
    #            tmpdata=data[event_index:upper,:,:,:].astype("float32")
    #            tmplabel=label[event_index:upper].astype("int32")
    #            tmpweight=weight[event_index:upper].astype("float32")
    #            tmpnweight=nweight[event_index:upper].astype("float32")
    #            
    #            if step_size<self.batch_size:
    #                d["hist"]=np.concatenate([d["hist"],tmpdata],axis=0)
    #                d["y"]=np.concatenate([d["y"],tmplabel],axis=0)
    #                d["weight"]=np.concatenate([d["weight"],tmpweight],axis=0)
    #                d["normalized_weight"]=np.concatenate([d["normalized_weight"],tmpnweight],axis=0)
    #            else:
    #                d["hist"]=tmpdata
    #                d["y"]=tmplabel
    #                d["weight"]=tmpweight
    #                d["normalized_weight"]=tmpnweight
    #            
    #            if d["hist"].shape[0]<self.batch_size:
    #               step_size=self.batch_size-d["hist"].shape[0]
    #               continue
    #            else:
    #                step_size=self.batch_size
    #                yield d



if __name__=="__main__":
    #run_split()
    mainpath='/global/cscratch1/sd/tkurth/atlas_dl/data_delphes'
    trainfiles=[mainpath+'/'+x for x in os.listdir(mainpath) if x.startswith('hep_training_')]
    validationfiles=[mainpath+'/'+x for x in os.listdir(mainpath) if x.startswith('hep_validation_')]
    testfiles=[mainpath+'/'+x for x in os.listdir(mainpath) if x.startswith('hep_test_')]

