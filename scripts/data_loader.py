
import matplotlib; matplotlib.use("agg")


__author__ = 'racah'
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import sys
import os
import time
import h5py

#sys.path.append('/global/homes/w/wbhimji/cori-envs/nersc-rootpy/lib/python2.7/site-packages/')



def shuffle(x,y):
    inds = np.arange(x.shape[0])

    #shuffle data
    rng = np.random.RandomState(7)
    rng.shuffle(inds)
    return x[inds], y[inds]

def split_train_val(x,y, prop):
    tr_prop = 1 - prop
    inds = np.arange(x.shape[0])
    #split train, val, test
    tr_inds = inds[:int((tr_prop*len(inds)))]
    val_inds = inds[int(tr_prop*len(inds)):]

    x_tr, y_tr, x_val, y_val = x[tr_inds], y[tr_inds], x[val_inds], y[val_inds]
    return x_tr, y_tr, x_val, y_val



def preprocess(x, max_abs=None):
    '''a type of sparse preprocessing, which scales everything between -1 and 1 without losing sparsity'''
    #only calculate the statistic using training set
    if max_abs is None:
        max_abs=np.abs(x).max(axis=(0,1,2,3))

    #then scale all sets
    x /= max_abs
    #print np.max(x)
    #print np.min(x)
    return x, max_abs



# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])



class DataLoader(object):
    def __init__(self, bg_cfg_file = './config/BgFileListAug16.txt',
                sig_cfg_file='./config/SignalFileListAug16.txt',
                type_ = "hdf5",
                num_events=50000,
                preprocess=True,
                bin_size=0.025,
                eta_range = [-5,5],
                phi_range = [-3.14, 3.14], val_prop=0.2, use_premade=False):
        
        
        self.bg_files = bg_cfg_file if isinstance(bg_cfg_file, list) else [bg_cfg_file]
        self.sig_files = sig_cfg_file if isinstance(sig_cfg_file, list) else [sig_cfg_file]
        self.val_prop = val_prop
        self.phi_bins = int(np.floor((phi_range[1] - phi_range[0]) / bin_size))
        self.eta_bins = int(np.floor((eta_range[1] - eta_range[0]) / bin_size))
        self.eta_range = eta_range
        self.phi_range = phi_range
        #we assume there are more bg per file than sig, so we bound our number of files by number of files
        #needed for a sig event
        assert num_events % 2 == 0, "why an odd number for num_events?!, even please"
        self.num_each = num_events / 2

        self.use_premade = use_premade
        self.type_ = type_
        
        if type_ == "xaod" or type_ == "delphes":
            self.file_type = "root"
            import ROOT
            import rootpy
            import root_numpy as rnp
            self.set_root_cfgs()
        else:
            self.file_type = "hdf5"
            self.set_h5_cfgs()
            
    def set_h5_cfgs(self):
        self.group_prefix = "event_"
        self.h5keys = ['clusE',
         'clusEta',
         'clusPhi']
         
                       
#                        u'hist',
#          u'mGlu',
#          u'mNeu',
#          u'passSR',
#          u'weight']
        
        

    def set_data_cfgs(self):
        self.events_per_sig_file = 10000
        self.bg_files = [line.rstrip() for line in open(self.bg_cfg_files)]
        self.sig_files = [line.rstrip() for line in open(self.sig_cfg_files)]
        #get the number of files needed
        num_files = int(np.ceil(self.num_each / float(self.events_per_sig_file)))

        #because root does not do well with one file
        self.num_files = num_files if num_files > 1 else 2
        
        

        if self.type_ == "delphes":
            self.branchMap = {
               'Tower.Eta' : 'ClusEta',
               'Tower.Phi' : 'ClusPhi',
               'Tower.E' : 'ClusE',
               'FatJet.PT' : 'FatJetPt',
               'FatJet.Eta' : 'FatJetEta',
               'FatJet.Phi' : 'FatJetPhi',
               'FatJet.Mass' : 'FatJetM',
            }

            self.treename = 'Delphes'
        elif self.type_ == "xaod":
            


            self.branch_map = {
                            'CaloCalTopoClustersAuxDyn.calEta' : 'ClusEta',
                            'CaloCalTopoClustersAuxDyn.calPhi' : 'ClusPhi',
                            'CaloCalTopoClustersAuxDyn.calE' : 'ClusE',
                            'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.pt' : 'FatJetPt',
                            'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.eta' : 'FatJetEta',
                            'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.phi' : 'FatJetPhi',
                            'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.m' : 'FatJetM',
                        }

            self.treename='CollectionTree'
    

    def grab_events(self, file_list, num_files_or_events, start=0):
        if self.file_type == "root":
            x = self._grab_root_events(file_list, num_files_or_events,start)
        else:
            X=[]
            for file_ in file_list:
                x = self._grab_hdf5_events(file_, num_files_or_events / len(file_list), start)
                X.append(x)
            x = np.vstack(tuple(X))
        return x
            
    
    def _grab_hdf5_events(self,file_, num_events, start):
        h5f = h5py.File(file_)
        
        if self.use_premade:
            x = np.zeros((num_events, 1, 50,50 ))
        else:
            x = np.zeros((num_events, 1, self.phi_bins, self.eta_bins ))
        for cnt, i in enumerate(range(start, num_events)):
            event = h5f[self.group_prefix + str(i)]
            if self.use_premade:
                x[cnt][0] = event["hist"][:]
            else:
                #print event.keys()
                d = {k.lower():event[k] for k in self.h5keys }
                x[cnt][0] = self.make_hist(d)
        return x
        
        
        
    def _grab_root_events(self,file_list, num_files, start=0):
        #so we don't have annoying stderr messages
        with suppress_stdout_stderr():

            #bgarray has n_events groups of 3 parallel numpy arrays 
            #(each numpy within a group is of equal length and each array corresponds to phi, eta and the corresponding energy)
            array = rnp.root2array(file_list[:num_files],
                                     treename=self.treename,
                                     branches=self.branch_map.keys(),
                                     start=start,
                                     stop=self.num_each,
                                     warn_missing_tree=True)




        x = np.zeros((self.num_each, 1, self.phi_bins, self.eta_bins ))

        for i in range(self.num_each):
            d = {v.lower() : array[k][i] for k,v in branch_map.iteritems()}
            x[i][0] = self.make_hist(d) 
            
        return x
            
    def make_hist(self, d):
        
        return np.histogram2d(d['clusphi'],d['cluseta'], bins=(self.phi_bins, self.eta_bins),
                              weights=d["cluse"], range=[self.phi_range,self.eta_range])[0] 

   


    def load_data(self):
        num = self.num_files if self.file_type == "root" else self.num_each
        x_bg = self.grab_events(self.bg_files, num)
        x_sig = self.grab_events(self.sig_files, num)

        #background first
        x = np.vstack((x_bg, x_sig))

        # 1 means signal, 0 means background
        y = np.zeros((2*self.num_each,)).astype('int32')
        #make the last half signal label
        y[self.num_each:] = 1
        
        x,y = shuffle(x,y)
        xt,yt,xv,yv = split_train_val(x,y, self.val_prop)
        xt,tm = preprocess(xt)
        xv, _ = preprocess(xv,tm)
        return xt, yt, xv, yv 
    
    def iterate_data(self, batch_size=128):
#         if self.num_each < batch_size / 2:
#             batch_size = 2 * self.num_each
#         #only support for hdf5
#         for i in range(0, self.num_each, batch_size / 2):
#             x_bg = self.grab_events(self.bg_files, batch_size / 2, i)
#             x_sig = self.grab_events(self.sig_files, batch_size / 2, i)
        pass



if __name__=="__main__":
    dl = DataLoader(bg_cfg_file = '../config/BgFileListAug16.txt',
                sig_cfg_file='../config/SignalFileListAug16.txt',)









