
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
    def __init__(self, filelist, batch_size=128, shuffle=True, allow_fractional_batches=False, keys=["data","label","weight","normweight"]):
        #keys
        self.keys=keys
        #batchsize and indices
        self.batch_size=batch_size
        #store the filelist
        self.files=filelist
        self.num_files=len(self.files)
        #store the shuffle state
        self.shuffle=shuffle
        #allow fractional batches?
        self.allow_fractional_batches=allow_fractional_batches
        #file and event indices:
        self.file_index=0
        self.event_index=0
        #hgroup:
        self.hgroup={}
        
        #determine how many events we have:
        self.num_events=0
        for fname in self.files:
            f=h5py.File(fname,'r')
            count=f[self.keys[0]].shape[0]
            f.close()
            self.num_events+=count
        
        #shuffle files
        if self.shuffle:
            np.random.shuffle(self.files)
            
        #load the initial bunch of data
        self.load_next_file()
    
    
    #iterator
    def __iter__(self):
        #shuffle if requested
        if self.shuffle:
            np.random.shuffle(self.files)
        #reset counters
        self.file_index=0
        self.event_index=0
        #prefetch next
        self.load_next_file()
        #and go:
        return self
    
    
    #load next file logic
    def load_next_file(self):
        #open file
        f=h5py.File(self.files[self.file_index],'r')
        #load data from file
        for key in self.keys:
            self.hgroup[key]=f[key].value
        #close file
        f.close()
        
        #datalength:
        self.dlength=self.hgroup[self.keys[0]].shape[0]
        
        #shuffle data if requested
        if self.shuffle:
            reindex=np.random.permutation(self.dlength)
            for key in self.keys:
                self.hgroup[key]=self.hgroup[key][reindex]
    
    
    #next function
    def __next__(self):
        #stop the iteration here
        if self.file_index>=self.num_files:
            raise StopIteration
        
        #grep data
        #upper index
        upper=np.min([self.dlength,self.event_index+self.batch_size])
        #load data
        tmphgroup={}
        for key in self.keys:
            tmphgroup[key]=self.hgroup[key][self.event_index:upper]

        #load new file if needed:
        if self.dlength<=(self.event_index+self.batch_size):
            self.file_index+=1
            
            #check if the epoch is over
            if self.file_index>=self.num_files:
                #return the remainder
                if self.allow_fractional_batches:
                    return tmphgroup
            else:
                #prefetch the file
                self.load_next_file()
                #fetch the missing data:
                rlength=self.batch_size-tmphgroup[self.keys[0]].shape[0]
                for key in self.keys:
                    tmphgroup[key]=np.concatenate([tmphgroup[key],self.hgroup[key][0:rlength]],axis=0)
                self.event_index=rlength
        else:
            self.event_index+=self.batch_size
        
        #stop the iteration here
        if self.file_index>=self.num_files:
            raise StopIteration
        
        #return result
        return tmphgroup
    
    
    #backwards compatibility
    def next(self):
        return self.__next__()
    
    
    #returns all the data in one big dictionary. HANDLE WITH CARE, it can easily overflow memory!
    def get_all(self):
        result={}
        
        #load first
        f=h5py.File(self.files[0],'r')
        for key in self.keys:
            result[key]=f[key].value
        f.close()
        #load the rest
        for fname in self.files[1:]:
            f=h5py.File(fname,'r')
            for key in self.keys:
                result[key]=np.concatenate([result[key],f[key].value])
            f.close()
        #return result
        return result



if __name__=="__main__":
    #set the filelists
    mainpath='/global/cscratch1/sd/tkurth/atlas_dl/data_delphes'
    trainfiles=[mainpath+'/'+x for x in os.listdir(mainpath) if x.startswith('hep_training_')]
    validationfiles=[mainpath+'/'+x for x in os.listdir(mainpath) if x.startswith('hep_validation_')]
    testfiles=[mainpath+'/'+x for x in os.listdir(mainpath) if x.startswith('hep_test_')]

