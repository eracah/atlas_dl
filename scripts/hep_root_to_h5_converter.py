
import matplotlib; matplotlib.use("agg")


__author__ = 'tkurth'
import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import time
import re
sys.path.append('/global/homes/w/wbhimji/cori-envs/nersc-rootpy/lib/python2.7/site-packages/')
sys.path.append('/global/common/cori/software/root/6.06.06/lib/root')
import ROOT
import rootpy
import root_numpy as rnp
import h5py as h5


# ## Useful functions


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



def merge_dicts(dict1,dict2):
    tmp = dict1.copy()
    tmp.update(dict2)
    return tmp



#file string parser
def parse_filename(fname,directory='.'):
    directory=re.sub(r'^(.*?)(/+)$',r'\1',directory)
    
    #signal file?
    smatch=re.compile(r'GG_RPV(.*?)_(.*?)_(.*?)\.merge')
    tmpres=smatch.findall(fname)
    if tmpres:
        tmpres=tmpres[0]
        return {'rpv':int(tmpres[0]), 'mass1':int(tmpres[1]), 'mass2':int(tmpres[2]), 'name':directory+'/'+fname}

    #background file?
    smatch=re.compile(r'JZ(.*?)\.merge')
    tmpres=smatch.findall(fname)
    if tmpres:
        return {'jz':int(tmpres[0]), 'name':directory+'/'+fname}

    #nothing at all
    return {}



def load_data(filelists,
                group_name='CollectionTree',
                branches=['CaloCalTopoClustersAuxDyn.calPhi', \
                          'CaloCalTopoClustersAuxDyn.calEta', \
                          'CaloCalTopoClustersAuxDyn.calE'],
                dataset_name='histo',
                type_='root'):
    
    #iterate over elements in the filelists
    records=[]
    for fname in filelists:
        #read specifics of that list
        masterrec=parse_filename(fname.split('/')[-1])
        #determine if it is label or background
        if 'jz' in masterrec.keys():
            masterrec['label']=0
        else:
            masterrec['label']=1
        
        #read the files in the filelist
        files = [line.rstrip() for line in open(fname)]
        
        #we don't want annoying stderr messages
        with suppress_stdout_stderr():
            
            #bgarray has n_events groups of 3 parallel numpy arrays 
            #(each numpy within a group is of equal length and each array corresponds to phi, eta and the corresponding energy)
            try:
                datarec = rnp.root2array(files,                                         treename=group_name,                                         branches=branches,                                         start=0,                                         warn_missing_tree=True)
                tmpdf=pd.DataFrame.from_records(datarec)
                reclist=tmpdf[['CaloCalTopoClustersAuxDyn.calPhi',                                 'CaloCalTopoClustersAuxDyn.calEta',                                 'CaloCalTopoClustersAuxDyn.calE']].to_dict('records')
                reclist=[merge_dicts(masterrec,rec) for rec in reclist]
                    
            except:
                continue
            
        #append to records
        records+=reclist
            
    #return dataframe
    return pd.DataFrame(records)


#preprocessor
def preprocess_data(df,eta_range,phi_range,eta_bins,phi_bins):
    #empty array
    xvals = np.zeros((df.shape[0], 1, phi_bins, eta_bins ),dtype='float32')
    yvals = np.zeros((df.shape[0],),dtype='int32')
    
    for i in range(df.shape[0]):        
        phi, eta, E =  df.iloc[i]['CaloCalTopoClustersAuxDyn.calPhi'],                       df.iloc[i]['CaloCalTopoClustersAuxDyn.calEta'],                       df.iloc[i]['CaloCalTopoClustersAuxDyn.calE']
        
        xvals[i]=np.histogram2d(phi,eta,
                                bins=(phi_bins, eta_bins), \
                                weights=E,
                                range=[phi_range,eta_range])[0]
        yvals[i]=df.iloc[i]['label']
        
    return xvals, yvals



class hep_data_iterator:
    
    #class constructor
    def __init__(self,
                 datadf,
                 max_frequency=None,
                 even_frequencies=True,
                 shuffle=True,
                 nbins=(100,100),
                 eta_range = [-5,5],
                 phi_range = [-3.1416, 3.1416]
                ):

        #set parameters
        self.shuffle = shuffle
        self.nbins = nbins
        self.eta_range = eta_range
        self.phi_range = phi_range
        
        #even frequencies?
        self.even_frequencies=even_frequencies
        
        #compute bins depending on total range
        eta_step=(self.eta_range[1]-self.eta_range[0])/(self.nbins[0]-1)
        self.eta_bins = np.arange(self.eta_range[0],self.eta_range[1]+eta_step,eta_step)
        phi_step=(self.phi_range[1]-self.phi_range[0])/(self.nbins[1]-1)
        self.phi_bins = np.arange(self.phi_range[0],self.phi_range[1]+phi_step,phi_step)
        
        #dataframe
        self.df = datadf
        self.df.sort_values(by='label',inplace=True)
        
        #make class frequencies even:
        tmpdf=self.df.groupby('label').count().reset_index()
        self.num_classes=tmpdf.shape[0]
        
        #determine minimum frequency
        min_frequency=tmpdf['CaloCalTopoClustersAuxDyn.calE'].min()
        if max_frequency:
            min_frequency=np.min([min_frequency,max_frequency])
        elif not self.even_frequencies:
            min_frequency=-1
        
        tmpdf=self.df.groupby(['label']).apply(lambda x: x[['CaloCalTopoClustersAuxDyn.calPhi',                                                             'CaloCalTopoClustersAuxDyn.calEta',                                                             'CaloCalTopoClustersAuxDyn.calE']].iloc[:min_frequency,:]).copy()
        tmpdf.reset_index(inplace=True)
        del tmpdf['level_1']
        
        #copy tmpdf into self.df:
        self.df=tmpdf.copy()
        
        #compute max:
        self.compute_data_max()
        
        #shuffle if wanted (highly recommended)
        if self.shuffle:
            self.df=self.df.reindex(np.random.permutation(self.df.index))
        
        #number of examples
        self.num_examples=self.df.shape[0]
        
        #shapes:
        self.xshape=(1, self.phi_bins, self.eta_bins)
        
    
    #compute max over all data
    def compute_data_max(self):
        '''compute the maximum over all event entries for rescaling data between -1 and 1'''
        self.max_abs=(self.df['CaloCalTopoClustersAuxDyn.calE'].abs()).apply(lambda x: np.max(x)).max()
    
    
    #this is the batch iterator:
    def next_batch(self,batchsize):
        '''batch iterator'''
        
        #shuffle:
        if self.shuffle:
            self.df=self.df.reindex(np.random.permutation(self.df.index))
        
        #iterate
        for idx in range(0,self.num_examples-batchsize,batchsize):
            #yield next batch
            x,y=preprocess_data(self.df.iloc[idx:idx+batchsize,:],                             self.eta_range,
                             self.phi_range,
                             self.eta_bins,self.phi_bins)
            #rescale x:
            x/=self.max_abs
        
            #return result
            yield x,y


# ## Curate file list


directory='/project/projectdirs/das/wbhimji/RPVSusyJetLearn/atlas_dl/config/'
filelists=[parse_filename(x,directory) for x in os.listdir(directory) if x.startswith('mc')]
filenamedf=pd.DataFrame(filelists)


# ## Select signal configuration


#select signal configuration
mass1=1400
mass2=850
sig_cfg_files=list(filenamedf[ (filenamedf['mass1']==1400) & (filenamedf['mass2']==850) ]['name'])

#select background configuration
jzmin=4
jzmax=5
bg_cfg_files=list(filenamedf[ (filenamedf['jz']>=jzmin) & (filenamedf['jz']<=jzmax) ]['name'])


# ## Load data


#load background files
bgdf=load_data(bg_cfg_files)
bgdf=bgdf.reindex(np.random.permutation(bgdf.index))



#load signal data
sigdf=load_data(sig_cfg_files)
sigdf=sigdf.reindex(np.random.permutation(sigdf.index))



#parameters
train_fraction=0.75
validation_fraction=0.003
nbins=(227,227)

#create sizes:
num_sig_train=int(np.floor(sigdf.shape[0]*train_fraction))
#num_bg_train=int(np.floor(bgdf.shape[0]*train_fraction))
num_bg_train=num_sig_train
num_sig_validation=int(np.floor(sigdf.shape[0]*validation_fraction))
num_bg_validation=int(np.floor(bgdf.shape[0]*validation_fraction))
#num_bg_validation=num_sig_validation

#split the sets
traindf=pd.concat([bgdf.iloc[:num_bg_train],sigdf.iloc[:num_sig_train]])
validdf=pd.concat([bgdf.iloc[num_bg_train:num_bg_train+num_bg_validation],                    sigdf.iloc[num_sig_train:num_sig_train+num_sig_validation]])
testdf=pd.concat([bgdf.iloc[num_bg_train+num_bg_validation:],                    sigdf.iloc[num_sig_train+num_sig_validation:]])

#create iterators
hditer_train=hep_data_iterator(traindf,nbins=nbins)
hditer_validation=hep_data_iterator(validdf,nbins=nbins,even_frequencies=False)
hditer_test=hep_data_iterator(testdf,nbins=nbins,even_frequencies=False)

#the preprocessing for the validation iterator has to be taken from the training iterator
hditer_validation.max_abs=hditer_train.max_abs
hditer_test.max_abs=hditer_train.max_abs


# ## Preprocess Data


datadir="/global/cscratch1/sd/tkurth/atlas_dl/data"
numnodes=9300



#print ensemble sizes and determine the chunk size
chunksize_train=int(np.ceil(hditer_train.num_examples/numnodes))
print hditer_train.num_examples, chunksize_train
chunksize_validation=int(np.ceil(hditer_validation.num_examples/numnodes))
print hditer_validation.num_examples, chunksize_validation
chunksize_test=np.min([int(np.ceil(hditer_test.num_examples/numnodes)),60000])
print hditer_test.num_examples, chunksize_test


# ### Training


for idx,i in enumerate(range(0,hditer_train.num_examples,chunksize_train)):
    iup=np.min([i+chunksize_train,hditer_train.num_examples])
    
    #preprocess
    x,y=preprocess_data(hditer_train.df.iloc[i:iup],                         hditer_train.eta_range,                         hditer_train.phi_range,                         hditer_train.eta_bins,                         hditer_train.phi_bins)
    x/=hditer_train.max_abs
    
    #write file
    f = h5.File(datadir+'/hep_training_chunk'+str(idx)+'.hdf5','w')
    f['data']=x
    f['label']=y
    f.close()


# ### Test


#chunk it to fit it into memory
for idx,i in enumerate(range(0,hditer_test.num_examples,chunksize_test)):
    iup=np.min([i+chunksize_test,hditer_test.num_examples])
    
    #preprocess
    x,y=preprocess_data(hditer_test.df.iloc[i:iup],                         hditer_test.eta_range,                         hditer_test.phi_range,                         hditer_test.eta_bins,                         hditer_test.phi_bins)
    x/=hditer_train.max_abs
    
    #write file
    f = h5.File(datadir+'/hep_test_chunk'+str(idx)+'.hdf5','w')
    f['data']=x
    f['label']=y
    f.close()


# ### Validation


for idx,i in enumerate(range(0,hditer_validation.num_examples,chunksize_validation)):
    iup=np.min([i+chunksize_validation,hditer_validation.num_examples])
    
    #preprocess
    x,y=preprocess_data(hditer_validation.df.iloc[i:iup],                     hditer_validation.eta_range,                     hditer_validation.phi_range,                     hditer_validation.eta_bins,                     hditer_validation.phi_bins)
    x/=hditer_train.max_abs
    
    #write the file
    f = h5.File(datadir+'/hep_validation_chunk'+str(idx)+'.hdf5','w')
    f['data']=x
    f['label']=y
    f.close()





