
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
#sys.path.append('/global/homes/w/wbhimji/cori-envs/nersc-rootpy/lib/python2.7/site-packages/')
#sys.path.append('/global/common/cori/software/root/6.06.06/lib/root')
#import ROOT
#import rootpy
#import root_numpy as rnp
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
    smatch=re.compile(r'^GG_RPV(.*?)_(.*?)_(.*?)\.h5')
    tmpres=smatch.findall(fname)
    if tmpres:
        tmpres=tmpres[0]
        return {'rpv':int(tmpres[0]), 'mass1':int(tmpres[1]), 'mass2':int(tmpres[2]), 'name':directory+'/'+fname}

    #background file?
    smatch=re.compile(r'^jetjet_JZ(.*?)\.h5')
    tmpres=smatch.findall(fname)
    if tmpres:
        return {'jz':int(tmpres[0]), 'name':directory+'/'+fname}

    #nothing at all
    return {}



def load_data(filelists,
                group_name='CollectionTree',
                branches=['clusPhi',
                          'clusEta',
                          'clusE',
                         'weight',
                         'passSR'],
                dataset_name='histo',
                type_='hdf5'):
    
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
        
        #open the hdf5 file
        #we don't want annoying stderr messages
        try:
            reclist=[]
            f= h5.File(fname,'r')
            for event in f.items():
                if event[0].startswith('event'):
                    datarec={}
                    
                    datarec['CaloCalTopoClustersAuxDyn.calPhi']=event[1][branches[0]].value
                    datarec['CaloCalTopoClustersAuxDyn.calEta']=event[1][branches[1]].value
                    datarec['CaloCalTopoClustersAuxDyn.calE']=event[1][branches[2]].value
                    datarec['CaloCalTopoClustersAuxDyn.weight']=event[1][branches[3]].value
                    datarec['CaloCalTopoClustersAuxDyn.passSR']=event[1][branches[4]].value
                    if masterrec['label']==1:
                        datarec['CaloCalTopoClustersAuxDyn.mGlu']=event[1]['mGlu'].value
                        datarec['CaloCalTopoClustersAuxDyn.mNeu']=event[1]['mNeu'].value
                    else:
                        datarec['CaloCalTopoClustersAuxDyn.mGlu']=0.
                        datarec['CaloCalTopoClustersAuxDyn.mNeu']=0.
                    
                    reclist.append(merge_dicts(masterrec,datarec))
            
            #close file
            f.close()
            
        except:
            continue
            
        #append to records
        records+=reclist
            
    #return dataframe
    return pd.DataFrame(records)


#data augmentation
def augment_data(xarr,roll_angle):
    #flip in x:
    if np.random.random_sample()>=0.5:
        xarr=np.fliplr(xarr)
    #flip in y:
    if np.random.random_sample()>=0.5:
        xarr=np.flipud(xarr)
    #roll in x with period 2pi/8
    randroll=np.random.randint(0,8,size=1)[0]
    #determine granularity:
    rollunit=randroll*roll_angle
    xarr=np.roll(xarr, shift=rollunit, axis=1)
    
    return xarr
    
    
#preprocessor
def preprocess_data(df,eta_range,phi_range,eta_bins,phi_bins):
    #empty array
    xvals = np.zeros((df.shape[0], 1, phi_bins, eta_bins ),dtype='float32')
    yvals = np.zeros((df.shape[0],),dtype='int32')
    wvals = np.zeros((df.shape[0],),dtype='float32')
    pvals = np.zeros((df.shape[0],),dtype='int32')
    mgvals = np.zeros((df.shape[0],),dtype='float32')
    mnvals = np.zeros((df.shape[0],),dtype='float32')
    
    for i in range(df.shape[0]):        
        phi, eta, E, w, psr, mg, mn =  df.iloc[i]['CaloCalTopoClustersAuxDyn.calPhi'],                                        df.iloc[i]['CaloCalTopoClustersAuxDyn.calEta'],                                        df.iloc[i]['CaloCalTopoClustersAuxDyn.calE'],                                        df.iloc[i]['CaloCalTopoClustersAuxDyn.weight'],                                        df.iloc[i]['CaloCalTopoClustersAuxDyn.passSR'],                                        df.iloc[i]['CaloCalTopoClustersAuxDyn.mGlu'],                                        df.iloc[i]['CaloCalTopoClustersAuxDyn.mNeu']
        
        xvals[i]=np.histogram2d(phi,eta,
                                bins=(phi_bins, eta_bins),
                                weights=E,
                                range=[phi_range,eta_range])[0]
        
        #obtain the rest
        wvals[i]=w
        pvals[i]=psr
        mgvals[i]=mg
        mnvals[i]=mn
        yvals[i]=df.iloc[i]['label']
        
    return xvals, yvals, wvals, pvals, mgvals, mnvals



class hep_data_iterator:
    
    #class constructor
    def __init__(self,
                 datadf,
                 max_frequency=None,
                 even_frequencies=True,
                 shuffle=True,
                 nbins=(100,100),
                 eta_range = [-5,5],
                 phi_range = [-3.1416, 3.1416],
                 augment=False
                ):

        #set parameters
        self.shuffle = shuffle
        self.nbins = nbins
        self.eta_range = eta_range
        self.phi_range = phi_range
        
        #even frequencies?
        self.even_frequencies=even_frequencies
        self.augment=augment
        
        #compute bins depending on total range
        #eta
        #eta_step=(self.eta_range[1]-self.eta_range[0])/float(self.nbins[0]-1)
        #self.eta_bins = np.arange(self.eta_range[0],self.eta_range[1]+eta_step,eta_step)
        self.eta_bins=self.nbins[0]
        #phi
        #phi_step=(self.phi_range[1]-self.phi_range[0])/float(self.nbins[1]-1)
        #self.phi_bins = np.arange(self.phi_range[0],self.phi_range[1]+phi_step,phi_step)
        self.phi_bins=self.nbins[1]
        
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
        
        tmpdf=self.df.groupby(['label']).apply(lambda x: x[['CaloCalTopoClustersAuxDyn.calPhi',
                                                            'CaloCalTopoClustersAuxDyn.calEta',
                                                            'CaloCalTopoClustersAuxDyn.calE',
                                                            'CaloCalTopoClustersAuxDyn.weight',
                                                            'CaloCalTopoClustersAuxDyn.passSR',
                                                            'CaloCalTopoClustersAuxDyn.mGlu',
                                                            'CaloCalTopoClustersAuxDyn.mNeu'
                                                           ]].iloc[:min_frequency,:]).copy()
        
        tmpdf.reset_index(inplace=True)
        del tmpdf['level_1']
        
        #copy tmpdf into self.df:
        self.df=tmpdf.copy()
        
        #compute max:
        self.compute_data_max()
        self.compute_weight_max()
        
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
        
    def compute_weight_max(self):
        '''compute the maximum over all event weight entries for rescaling data between 0 and 1. Take abs to be safe'''
        self.wmax=(self.df['CaloCalTopoClustersAuxDyn.weight'].abs()).apply(lambda x: np.max(x)).max()
    
    
    #get a random chunk: here, even_freq means even frequencies in chunk.
    def get_chunk(self,chunksize,even_freq=False):
        shuffledf=self.df.reindex(np.random.permutation(self.df.index))
        if not even_freq:
            return shuffledf.copy()
        else:
            #group by classes:
            chunksize_per_class=int(np.ceil(chunksize/self.num_classes))
            tmpdf=shuffledf.groupby('label').apply(lambda x: x.iloc[:chunksize_per_class,:]).copy()
            tmpdf.reset_index(drop=True,inplace=True)
            return tmpdf.reindex(np.random.permutation(tmpdf.index))
    
    
    #this is the batch iterator:
    def next_batch(self,batchsize):
        '''batch iterator'''
        
        #shuffle:
        if self.shuffle:
            self.df=self.df.reindex(np.random.permutation(self.df.index))
        
        #iterate
        for idx in range(0,self.num_examples-batchsize,batchsize):
            #yield next batch
            x,y,w,p=preprocess_data(self.df.iloc[idx:idx+batchsize,:],
                             self.eta_range,
                             self.phi_range,
                             self.eta_bins,self.phi_bins)
            #rescale x:
            x/=self.max_abs
        
            #return result
            yield x,y,w,p


# ## Curate file list


directory='/project/projectdirs/dasrepo/atlas_rpv_susy/hdf5/prod004_2016_11_30'
filelists=[parse_filename(x,directory) for x in os.listdir(directory) if x.endswith('h5')]
filenamedf=pd.DataFrame(filelists)


# ## Select signal configuration


#select signal configuration
#mass1=1400
#mass2=850
sig_cfg_files=list(filenamedf[ (filenamedf['mass1']>0.) & (filenamedf['mass2']>0.) ]['name'])
#sig_cfg_files=list(filenamedf['name'])

#select background configuration
jzmin=1
jzmax=11
bg_cfg_files=list(filenamedf[ (filenamedf['jz']>=jzmin) & (filenamedf['jz']<=jzmax) ]['name'])


# ## Load data


#load background files
bgdf=load_data(bg_cfg_files)
np.random.seed(13)
bgdf=bgdf.reindex(np.random.permutation(bgdf.index))



#load signal data
sigdf=load_data(sig_cfg_files)
np.random.seed(13)
sigdf=sigdf.reindex(np.random.permutation(sigdf.index))



#parameters
train_fraction=0.85
train_fit_fraction=0.3
validation_fraction=0.05
nbins=(224,224)

#create sizes:
#total
num_sig_total=sigdf.shape[0]
num_bg_total=bgdf.shape[0]
#training
num_bg_train=int(np.floor(bgdf.shape[0]*train_fraction))
#validation
num_bg_validation=int(np.floor(bgdf.shape[0]*validation_fraction))

#split the sets
#we need two training sets here, because we fit the distribution also:
traindf=bgdf.iloc[:int(np.floor(num_bg_train*(1.-train_fit_fraction)))]
traindf_fit=bgdf.iloc[int(np.floor(num_bg_train*(1.-train_fit_fraction))):num_bg_train]
validdf=bgdf.iloc[num_bg_train:num_bg_train+num_bg_validation]
testdf=bgdf.iloc[num_bg_train+num_bg_validation:]

#create iterators
hditer_train=hep_data_iterator(traindf,nbins=nbins,even_frequencies=False)
hditer_train_fit=hep_data_iterator(traindf_fit,nbins=nbins,even_frequencies=False)
hditer_validation=hep_data_iterator(validdf,nbins=nbins,even_frequencies=False)
hditer_test=hep_data_iterator(testdf,nbins=nbins,even_frequencies=False)
hditer_test_signal=hep_data_iterator(sigdf,nbins=nbins,even_frequencies=False)

#the preprocessing for the validation iterator has to be taken from the training iterator
hditer_validation.max_abs=hditer_train.max_abs
hditer_train_fit.max_abs=hditer_train.max_abs
hditer_test.max_abs=hditer_train.max_abs
hditer_test_signal.max_abs=hditer_train.max_abs


# ## Preprocess Data


datadir="/global/cscratch1/sd/tkurth/atlas_dl/data_preselect_autoencoder"
numnodes=9300



#print ensemble sizes and determine the chunk size
chunksize_train=int(np.ceil(hditer_train.num_examples/numnodes))
print "Training set: total = ", hditer_train.num_examples, " chunksize = ", chunksize_train

chunksize_train_fit=int(np.ceil(hditer_train_fit.num_examples/numnodes))
print "Fitting set: total = ", hditer_train_fit.num_examples, " chunksize = ", chunksize_train_fit

chunksize_validation=int(np.ceil(hditer_validation.num_examples/numnodes))
print "Validation set: total = ", hditer_validation.num_examples, " chunksize = ", chunksize_validation

chunksize_test=np.min([int(np.ceil(hditer_test.num_examples/numnodes)),60000])
print "Test-BG set: total = ", hditer_test.num_examples, " chunksize = ", chunksize_test

chunksize_test_signal=np.min([int(np.ceil(hditer_test_signal.num_examples/numnodes)),60000])
print "Test-SG set: total = ", hditer_test_signal.num_examples, " chunksize = ", chunksize_test_signal


# ### Training


for idx,i in enumerate(range(0,hditer_train.num_examples,chunksize_train)):
    iup=np.min([i+chunksize_train,hditer_train.num_examples])
    
    #preprocess
    x,y,w,p,g,n=preprocess_data(hditer_train.df.iloc[i:iup],                                 hditer_train.eta_range,                                 hditer_train.phi_range,                                 hditer_train.eta_bins,                                 hditer_train.phi_bins)
    x/=hditer_train.max_abs
    
    #write the file
    f = h5.File(datadir+'/hep_train_chunk'+str(idx)+'.hdf5','w')
    f['data']=x
    f['label']=y
    #reweight the weights for proper testing
    f['weight']=w/( np.float(hditer_train.num_examples)/np.float(num_bg_total) )
    #normalize those weights for training
    f['normweight']=w/hditer_train.wmax
    f['psr']=p
    f.close()


# ### Fitting


for idx,i in enumerate(range(0,hditer_train_fit.num_examples,chunksize_train_fit)):
    iup=np.min([i+chunksize_train_fit,hditer_train_fit.num_examples])
    
    #preprocess
    x,y,w,p,g,n=preprocess_data(hditer_train_fit.df.iloc[i:iup],                                 hditer_train_fit.eta_range,                                 hditer_train_fit.phi_range,                                 hditer_train_fit.eta_bins,                                 hditer_train_fit.phi_bins)
    x/=hditer_train.max_abs
    
    #write the file
    f = h5.File(datadir+'/hep_trainfit_chunk'+str(idx)+'.hdf5','w')
    f['data']=x
    f['label']=y
    #reweight the weights for proper testing
    f['weight']=w/( np.float(hditer_train_fit.num_examples)/np.float(num_bg_total) )
    #normalize those weights for training
    f['normweight']=w/hditer_train.wmax
    f['psr']=p
    f.close()


# ### Test


#test for background
for idx,i in enumerate(range(0,hditer_test.num_examples,chunksize_test)):
    iup=np.min([i+chunksize_test,hditer_test.num_examples])
    
    #preprocess
    x,y,w,p,g,n=preprocess_data(hditer_test.df.iloc[i:iup],                                 hditer_test.eta_range,                                 hditer_test.phi_range,                                 hditer_test.eta_bins,                                 hditer_test.phi_bins)
    x/=hditer_train.max_abs
    
    #write file
    f = h5.File(datadir+'/hep_test_chunk'+str(idx)+'.hdf5','w')
    f['data']=x
    f['label']=y
    #reweight the weights for proper testing. This time with respect to combined signal and bg test sets
    f['weight']=w/( np.float(hditer_test.num_examples)/np.float(num_bg_total) )
    f['psr']=p
    f['mg']=g
    f['mn']=n
    f.close()



#test for signal
for idx,i in enumerate(range(0,hditer_test_signal.num_examples,chunksize_test_signal)):
    iup=np.min([i+chunksize_test_signal,hditer_test_signal.num_examples])
    
    #preprocess
    x,y,w,p,g,n=preprocess_data(hditer_test_signal.df.iloc[i:iup],                                 hditer_test_signal.eta_range,                                 hditer_test_signal.phi_range,                                 hditer_test_signal.eta_bins,                                 hditer_test_signal.phi_bins)
    x/=hditer_train.max_abs
    
    #write file
    f = h5.File(datadir+'/hep_test_signal_chunk'+str(idx)+'.hdf5','w')
    f['data']=x
    f['label']=y
    #we take everything so no reweighting needed:
    f['weight']=w
    f['psr']=p
    f['mg']=g
    f['mn']=n
    f.close()


# ### Validation


for idx,i in enumerate(range(0,hditer_validation.num_examples,chunksize_validation)):
    iup=np.min([i+chunksize_validation,hditer_validation.num_examples])
    
    #preprocess
    x,y,w,p,g,n=preprocess_data(hditer_validation.df.iloc[i:iup],                                 hditer_validation.eta_range,                                 hditer_validation.phi_range,                                 hditer_validation.eta_bins,                                 hditer_validation.phi_bins)
    x/=hditer_train.max_abs
    
    #write the file
    f = h5.File(datadir+'/hep_validation_chunk'+str(idx)+'.hdf5','w')
    f['data']=x
    f['label']=y
    f['weight']=w/( np.float(hditer_validation.num_examples)/np.float(num_bg_total) )
    f['normweight']=w/hditer_train.wmax
    f['psr']=p
    f['mg']=g
    f['mn']=n
    f.close()





