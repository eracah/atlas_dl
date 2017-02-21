
import matplotlib; matplotlib.use("agg")


__author__ = 'tkurth'
import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mlines
import matplotlib.font_manager as font_manager
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
import time
from tqdm import tqdm
import re
#sys.path.append('/global/homes/w/wbhimji/cori-envs/nersc-rootpy/lib/python2.7/site-packages/')
#sys.path.append('/global/common/cori/software/root/6.06.06/lib/root')
#import ROOT
#import rootpy
#import root_numpy as rnp
import h5py as h5


# # Plot Settings


# Set the font dictionaries (for plot title and axis titles)
title_font = {'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'size':'14'}
font_prop_title = font_manager.FontProperties(size=40)
font_prop_axis = font_manager.FontProperties(size=30)
font_prop_axis_labels = font_manager.FontProperties(size=40)
font_prop_legend = font_manager.FontProperties(size=26)
#plt.style.use('seaborn-talk')
#plt.style.available


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
        return {'rpv':int(tmpres[0]), 
                'mGlu':int(tmpres[1]), 
                'mNeu':int(tmpres[2]), 
                'jz': 0, 
                'filename': directory+'/'+fname}

    #background file?
    smatch=re.compile(r'^jetjet_JZ(.*?)\.h5')
    tmpres=smatch.findall(fname)
    if tmpres:
        return {'rpv': 0., 
                'mGlu': 0.,
                'mNeu': 0., 
                'jz': int(tmpres[0]), 
                'filename': directory+'/'+fname}

    #nothing at all
    return {}



def load_data(filelists,
                group_name='CollectionTree',
                dataset_name='histo',
                type_='hdf5'):
    
    #iterate over elements in the filelists
    records=[]
    
    for fname in tqdm(filelists):
        #read specifics of that list
        infile=fname.split('/')[-1]
        masterrec=parse_filename(infile)
        #determine if it is label or background
        if masterrec['mGlu']>0 or masterrec['mNeu']>0:
            masterrec['label']=1
        else:
            masterrec['label']=0
        
        #open the hdf5 file
        #we don't want annoying stderr messages
        try:
            reclist=[]
            f= h5.File(fname,'r')
            for event in f.items():
                if event[0].startswith('event'):
                    datarec={}
                    
                    #event id:
                    datarec['eventid']=int(event[0].split('_')[1])
                    
                    #calorimeter:
                    #azimuth
                    datarec['calPhi']=event[1]['clusPhi'].value
                    #rapidity
                    datarec['calEta']=event[1]['clusEta'].value
                    #energy deposit
                    datarec['calE']=event[1]['clusE'].value
                    #EM fraction
                    datarec['calEM']=event[1]['clusEM'].value
                    
                    #tracks:
                    #azimuth
                    datarec['trackPhi']=event[1]['trackPhi'].value
                    #rapidity
                    datarec['trackEta']=event[1]['trackEta'].value
                    
                    #weight
                    datarec['weight']=event[1]['weight'].value
                    
                    #passes standard regression?
                    datarec['passSR']=event[1]['passSR'].value
                    
                    #SUSY theory masses
                    #if masterrec['label']==1:
                    #    datarec['mGlu']=event[1]['mGlu'].value
                    #    datarec['mNeu']=event[1]['mNeu'].value
                    #else:
                    #    datarec['mGlu']=0.
                    #    datarec['mNeu']=0.
                    
                    #append to master list
                    reclist.append(merge_dicts(masterrec,datarec))
            
            #close file
            f.close()
            
        except:
            continue
            
        #append to records
        records.append(pd.DataFrame(reclist))
            
    #return dataframe
    return pd.concat(records)


#data augmentation
def augment_data(xarr,roll_angle):
    #flip in x:
    if np.random.random_sample()>=0.5:
        for c in range(xarr.shape[0]):
            xarr[c,:,:]=np.fliplr(xarr[c,:,:])
    #flip in y:
    if np.random.random_sample()>=0.5:
        for c in range(xarr.shape[0]):
            xarr[c,:,:]=np.flipud(xarr[c,:,:])
    #roll in x with period 2pi/8
    randroll=np.random.randint(0,8,size=1)[0]
    #determine granularity:
    rollunit=randroll*roll_angle
    for c in range(xarr.shape[0]):
        xarr[c,:,:]=np.roll(xarr[c,:,:], shift=rollunit, axis=1)
    #return augmented array
    return xarr
    
    
#preprocessor
def preprocess_data(df,eta_range,phi_range,eta_bins,phi_bins):
    #empty array
    xvals  = np.zeros((df.shape[0], 3, phi_bins, eta_bins ),dtype='float32')
    yvals  = np.zeros((df.shape[0],),dtype='int32')
    eidvals = np.zeros((df.shape[0],),dtype='int32')
    wvals  = np.zeros((df.shape[0],),dtype='float32')
    pvals  = np.zeros((df.shape[0],),dtype='int32')
    mgvals = np.zeros((df.shape[0],),dtype='float32')
    mnvals = np.zeros((df.shape[0],),dtype='float32')
    jzvals = np.zeros((df.shape[0],),dtype='int32')
    
    for i in range(df.shape[0]):
        calPhi   = df.iloc[i]['calPhi']
        calEta   = df.iloc[i]['calEta']
        calE     = df.iloc[i]['calE']
        calEM    = df.iloc[i]['calEM']
        trackPhi = df.iloc[i]['trackPhi']
        trackEta = df.iloc[i]['trackEta']
        w        = df.iloc[i]['weight']
        psr      = df.iloc[i]['passSR']
        mg       = df.iloc[i]['mGlu']
        mn       = df.iloc[i]['mNeu']
        jz       = df.iloc[i]['jz']
        
        #data
        xvals[i,0,:,:]=np.histogram2d(calPhi,calEta,
                                        bins=(phi_bins, eta_bins),
                                        weights=calE,
                                        range=[phi_range,eta_range])[0]
        xvals[i,1,:,:]=np.histogram2d(calPhi,calEta,
                                        bins=(phi_bins, eta_bins),
                                        weights=calEM,
                                        range=[phi_range,eta_range])[0]
        xvals[i,2,:,:]=np.histogram2d(trackPhi,trackEta,
                                        bins=(phi_bins, eta_bins),
                                        range=[phi_range,eta_range])[0]
        
        #obtain the rest
        wvals[i]=w
        pvals[i]=psr
        mgvals[i]=mg
        mnvals[i]=mn
        jzvals[i]=jz
        yvals[i]=df.iloc[i]['label']
        eidvals[i]=df.iloc[i]['eventid']
        
    return xvals, yvals, wvals, pvals, mgvals, mnvals, jzvals, eidvals



class hep_data_iterator:
    
    #class constructor
    def __init__(self,
                 datadf,
                 max_frequency=None,
                 even_frequencies=True,
                 nbins=(100,100),
                 eta_range = [-5,5],
                 phi_range = [-3.1416, 3.1416],
                 augment=False,
                 compute_max=True
                ):

        #set parameters
        self.nbins = nbins
        self.eta_range = eta_range
        self.phi_range = phi_range

        #even frequencies?
        self.even_frequencies=even_frequencies
        self.augment=augment
        self.compute_max=compute_max
        
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
        min_frequency=tmpdf['calE'].min()
        if max_frequency:
            min_frequency=np.min([min_frequency,max_frequency])
        elif not self.even_frequencies:
            min_frequency=-1
        
        tmpdf=self.df.groupby(['label']).apply(lambda x: x[['calPhi',
                                                            'calEta',
                                                            'calE',
                                                            'calEM',
                                                            'trackPhi',
                                                            'trackEta',
                                                            'weight',
                                                            'passSR',
                                                            'mGlu',
                                                            'mNeu',
                                                            'jz',
                                                            'eventid'
                                                           ]].iloc[:min_frequency,:]).copy()
        
        tmpdf.reset_index(inplace=True)
        del tmpdf['level_1']
        
        #copy tmpdf into self.df:
        self.df=tmpdf.copy()
        
        #compute maxima:
        if self.compute_max:
            self.compute_data_max()
            self.compute_weight_max()
        
        #number of examples
        self.num_examples=self.df.shape[0]
        
        #shapes:
        self.xshape=(3, self.phi_bins, self.eta_bins)
    
    
    #shuffle data
    def shuffle(self, seed=None):
        if seed:
            np.random.seed(seed)
        self.df=self.df.reindex(np.random.permutation(self.df.index))
    
    
    #compute max over all data
    def compute_data_max(self):
        '''compute the maximum over all event entries for rescaling data between -1 and 1'''
        #initialize
        self.max_abs=np.zeros(3)
        #fill
        self.max_abs[0]=self.df[['calPhi','calEta','calE']].apply(lambda x: np.max(np.histogram2d(x['calPhi'],x['calEta'],
                                                                            bins=(self.phi_bins, self.eta_bins),
                                                                            weights=x['calE'],
                                                                            range=[self.phi_range,self.eta_range])[0]),
                                                                  axis=1).max()
        self.max_abs[1]=self.df[['calPhi','calEta','calEM']].apply(lambda x: np.max(np.histogram2d(x['calPhi'],x['calEta'],
                                                                            bins=(self.phi_bins, self.eta_bins),
                                                                            weights=x['calEM'],
                                                                            range=[self.phi_range,self.eta_range])[0]),
                                                                  axis=1).max()
        self.max_abs[2]=self.df[['trackPhi','trackEta']].apply(lambda x: np.max(np.histogram2d(x['trackPhi'],x['trackEta'],
                                                                            bins=(self.phi_bins, self.eta_bins),
                                                                            range=[self.phi_range,self.eta_range])[0]),
                                                                  axis=1).max()
    
    #compute maximum of weights
    def compute_weight_max(self):
        '''compute the maximum over all event weight entries for rescaling data between 0 and 1. Take abs to be safe'''
        self.wmax=(self.df['weight'].abs()).apply(lambda x: np.max(x)).max()


# ## Curate file list


directory='/project/projectdirs/dasrepo/atlas_rpv_susy/hdf5/delphes_002_2017_01_11'
filelists=[parse_filename(x,directory) for x in os.listdir(directory) if x.endswith('h5')]
filenamedf=pd.DataFrame(filelists)


# ## Select signal configuration


trainselect=[{'mGlu':1400, 'mNeu': 850}]



#select signal configuration
sig_cfg_files=[]
for item in trainselect:
    sig_cfg_files+=list(filenamedf[ (filenamedf['mGlu']==item['mGlu']) & (filenamedf['mNeu']==item['mNeu']) ]['filename'])

#select background configuration
jzmin=3
jzmax=11
bg_cfg_files=list(filenamedf[ (filenamedf['jz']>=jzmin) & (filenamedf['jz']<=jzmax) ]['filename'])



##load additional signal files:
#other_sig_cfg_files=list( filenamedf[ (filenamedf['mGlu']>0.) | (filenamedf['mNeu']>0.) ]['filename'])
#other_sig_cfg_files=[x for x in other_sig_cfg_files if x not in sig_cfg_files]


# ## Load data


#load background files
print("Loading background data.")
bgdf=load_data(bg_cfg_files)
#sort
bgdf.sort_values(by=['filename','eventid'],inplace=True)
bgdf.reset_index(drop=True,inplace=True)



#load signal data
print("Loading signal data.")
sigdf=load_data(sig_cfg_files)
#sort
sigdf.sort_values(by=['filename','eventid'],inplace=True)
sigdf.reset_index(drop=True,inplace=True)



##load additional signal data
#print("Loading remaining signal data.")
#othersigdf=load_data(other_sig_cfg_files)
##sort
#othersigdf.sort_values(by=['filename','eventid'],inplace=True)
#othersigdf.reset_index(drop=True,inplace=True)


# ## Parameters


#parameters
train_fraction=0.75
validation_fraction=0.05
nbins=(224,224)
#nbins=(64,64)
total_files_per_jz=640000
total_files_per_theory=640000
#this will yield even class frequencies, because there are 9 jz's:
nsig_augment=1


# ## Shuffle Training data


#do the shuffle
#background
np.random.seed(13)
bgdf=bgdf.reindex(np.random.permutation(bgdf.index))
#signal
np.random.seed(13)
sigdf=sigdf.reindex(np.random.permutation(sigdf.index))




print("Initial rescaling.")
bgdf['weight']/=np.float(total_files_per_jz)
sigdf['weight']/=np.float(total_files_per_theory)
#othersigdf['weight']/=np.float(total_files_per_theory)


# ## Determine counts and merge with dataframe


print("Determine frequencies.")

#background:
bggroup=bgdf.groupby(['jz'])
tmpdf=pd.DataFrame(bggroup['calE'].count())
tmpdf.reset_index(inplace=True)
tmpdf.rename(columns={'calE':'frequency'},inplace=True)
bgdf=bgdf.merge(tmpdf,on='jz',how='left')

#signal:
siggroup=sigdf.groupby(['mGlu','mNeu'])
tmpdf=pd.DataFrame(siggroup['calE'].count())
tmpdf.reset_index(inplace=True)
tmpdf.rename(columns={'calE':'frequency'},inplace=True)
sigdf=sigdf.merge(tmpdf,on=['mGlu','mNeu'],how='left')


# ## Split Ensemble


print("Split ensemble.")

#compute sizes:
#total
num_sig_total=sigdf.shape[0]
num_bg_total=bgdf.shape[0]

#group sigdf according to mGlu and mNeu:
siggroup=sigdf.groupby(['mGlu','mNeu'])
bggroup=bgdf.groupby(['jz'])


#training
#for signal, group according to masses and take the fraction for every theory:
trainsigdf=siggroup.apply(lambda x: x.iloc[:int(np.floor(x.shape[0]*train_fraction))])
trainsigdf.reset_index(drop=True,inplace=True)
#for background, group according to jz and take the fraction for every jz
trainbgdf=bggroup.apply(lambda x: x.iloc[:int(np.floor(x.shape[0]*train_fraction))])
trainbgdf.reset_index(drop=True,inplace=True)


#validation
valsigdf=siggroup.apply(lambda x: x.iloc[int(np.floor(x.shape[0]*train_fraction))
                                        :int(np.floor(x.shape[0]*train_fraction))+int(np.floor(x.shape[0]*validation_fraction))])
valsigdf.reset_index(drop=True,inplace=True)
#for background, group according to jz and take the fraction for every jz
valbgdf=bggroup.apply(lambda x: x.iloc[int(np.floor(x.shape[0]*train_fraction))
                                       :int(np.floor(x.shape[0]*train_fraction))+int(np.floor(x.shape[0]*validation_fraction))])
valbgdf.reset_index(drop=True,inplace=True)


#test
testsigdf=siggroup.apply(lambda x: x.iloc[int(np.floor(x.shape[0]*train_fraction))+int(np.floor(x.shape[0]*validation_fraction)):])
testsigdf.reset_index(drop=True,inplace=True)
#for background, group according to jz and take the fraction for every jz
testbgdf=bggroup.apply(lambda x: x.iloc[int(np.floor(x.shape[0]*train_fraction))+int(np.floor(x.shape[0]*validation_fraction)):])
testbgdf.reset_index(drop=True,inplace=True)


# ## Rescale the weights according to splits


#rescale the weights according to frequencies in mn/mg and jz combinations:
#training
traindf=pd.concat([trainbgdf,trainsigdf])
#traingroup=traindf.groupby(['jz','mGlu','mNeu'])
#tmpdf=pd.DataFrame(traingroup['calE'].count())
#tmpdf.reset_index(inplace=True)
#tmpdf.rename(columns={'calE':'split_frequency'},inplace=True)
#traindf=traindf.merge(tmpdf,on=['jz','mGlu','mNeu'],how='left')
#traindf['split_fraction']=traindf['split_frequency']/traindf['frequency']
#traindf['weight']/=traindf['split_fraction']


#validation
valdf=pd.concat([valbgdf,valsigdf])
#valgroup=valdf.groupby(['jz','mGlu','mNeu'])
#tmpdf=pd.DataFrame(valgroup['calE'].count())
#tmpdf.reset_index(inplace=True)
#tmpdf.rename(columns={'calE':'split_frequency'},inplace=True)
#valdf=valdf.merge(tmpdf,on=['jz','mGlu','mNeu'],how='left')
#valdf['split_fraction']=valdf['split_frequency']/valdf['frequency']
#valdf['weight']/=valdf['split_fraction']


#test
testdf=pd.concat([testbgdf,testsigdf])
#testgroup=testdf.groupby(['jz','mGlu','mNeu'])
#tmpdf=pd.DataFrame(testgroup['calE'].count())
#tmpdf.reset_index(inplace=True)
#tmpdf.rename(columns={'calE':'split_frequency'},inplace=True)
#testdf=testdf.merge(tmpdf,on=['jz','mGlu','mNeu'],how='left')
#testdf['split_fraction']=testdf['split_frequency']/testdf['frequency']
#testdf['weight']/=testdf['split_fraction']

#finally, append the other test: no need for rescaling here:
#testdf=pd.concat([testdf,othersigdf])


# ## Create Iterators


print("Create iterators.")

#create iterators
#training
hditer_train=hep_data_iterator(traindf,nbins=nbins,even_frequencies=False,augment=True,compute_max=True)
#shuffle this one
hditer_train.shuffle(13)

#validation
hditer_validation=hep_data_iterator(valdf,nbins=nbins,even_frequencies=False,compute_max=False)
#shuffle this one, so that I do not need to parse all validation files to find positive and negative examples
hditer_validation.shuffle(13)

#test
hditer_test=hep_data_iterator(testdf,nbins=nbins,even_frequencies=False,compute_max=False)
#shuffle this one, so that I do not need to parse all test files to find positive and negative examples
hditer_test.shuffle(13)


#the preprocessing for the validation iterator has to be taken from the training iterator
#validation
hditer_validation.max_abs=hditer_train.max_abs
hditer_validation.wmax=hditer_train.wmax
#test
hditer_test.max_abs=hditer_train.max_abs
hditer_test.wmax=hditer_train.wmax



print "Max Weight: ",hditer_train.df['weight'].max()
print "Min Weight: ",hditer_train.df['weight'].min()
print "Median Weight: ",hditer_train.df['weight'].median()
print "Mean Weight: ",hditer_train.df['weight'].mean()
print "Sum Weight: ",hditer_train.df['weight'].sum()


# ## Preprocess Data


datadir="/global/cscratch1/sd/tkurth/atlas_dl/data_delphes_new"
numnodes=1024



#print ensemble sizes and determine the chunk size
chunksize_train=int(np.ceil(2.*hditer_train.df.ix[ hditer_train.df.label==0 ].shape[0]/numnodes))
chunksize_train=int(np.floor(chunksize_train/(2*nsig_augment)))*2*nsig_augment
print "Training size: ",int(np.ceil(2.*hditer_train.df.ix[ hditer_train.df.label==0 ].shape[0])),' chunk size: ',chunksize_train
chunksize_validation=int(np.ceil(hditer_validation.num_examples/numnodes))
print "Validation size: ",hditer_validation.num_examples,' chunk size: ',chunksize_validation
chunksize_test=np.min([int(np.ceil(hditer_test.num_examples/numnodes)),60000])
print "Test size: ",hditer_test.num_examples,' chunk size: ',chunksize_test


# ### Training


print("Save training files.")

#here we have to treat background and signal separately
bgtrain=hditer_train.df.ix[ hditer_train.df.label==0 ]
sigtrain=hditer_train.df.ix[ hditer_train.df.label==1 ]
upper=int(np.floor(numnodes*chunksize_train/2))

for i in tqdm(range(0,numnodes)):
    
    #get background
    ilow=i*chunksize_train/2
    iup=np.min([(i+1)*chunksize_train/2,upper])
    #preprocess
    xbg,ybg,wbg,pbg,mgbg,mnbg,jzbg,eidbg = preprocess_data(bgtrain.iloc[ilow:iup],                                         hditer_train.eta_range,                                         hditer_train.phi_range,                                         hditer_train.eta_bins,                                         hditer_train.phi_bins)
    for c in range(3):
        xbg[:,c,:,:]/=hditer_train.max_abs[c]
    
    #get signal
    ilow=i*chunksize_train/(2*nsig_augment)
    iup=np.min([(i+1)*chunksize_train/(2*nsig_augment),upper])
    #preprocess
    xsg,ysg,wsg,psg,mgsg,mnsg,jzsg,eidsg = preprocess_data(sigtrain.iloc[ilow:iup],                                         hditer_train.eta_range,                                         hditer_train.phi_range,                                         hditer_train.eta_bins,                                         hditer_train.phi_bins)
    for c in range(3):
        xsg[:,c,:,:]/=hditer_train.max_abs[c]

    #tile the arrays
    xsg=np.tile(xsg,(nsig_augment,1,1,1))
    ysg=np.tile(ysg,(nsig_augment))
    wsg=np.tile(wsg,(nsig_augment))
    psg=np.tile(psg,(nsig_augment))
    mgsg=np.tile(mgsg,(nsig_augment))
    mnsg=np.tile(mnsg,(nsig_augment))
    jzsg=np.tile(jzsg,(nsig_augment))
    eidsg=np.tile(eidsg,(nsig_augment))
    #augment the x-values
    for k in range(0,xsg.shape[0]):
        xsg[k,:,:,:]=augment_data(xsg[k,:,:,:],int(np.round(hditer_train.phi_bins/8.)))
   
    #stack them together
    x=np.concatenate([xbg,xsg])
    y=np.concatenate([ybg,ysg])
    w=np.concatenate([wbg,wsg])
    p=np.concatenate([pbg,psg])
    mg=np.concatenate([mgbg,mgsg])
    mn=np.concatenate([mnbg,mnsg])
    jz=np.concatenate([jzbg,jzsg])
    eid=np.concatenate([eidbg,eidsg])
    
    #write file
    f = h5.File(datadir+'/hep_training_chunk'+str(i)+'.hdf5','w')
    f['data']=x
    f['label']=y
    f['weight']=w
    #normalize those weights for training
    f['normweight']=w/hditer_train.wmax
    f['psr']=p
    f['mg']=mg
    f['mn']=mn
    f['jz']=jz
    f['eid']=eid
    f.close()


# ### Test


print("Save test files.")

#chunk it to fit it into memory
for idx,i in tqdm(enumerate(range(0,hditer_test.num_examples,chunksize_test))):
    iup=np.min([i+chunksize_test,hditer_test.num_examples])
    
    #preprocess
    x,y,w,p,mg,mn,jz,eid = preprocess_data(hditer_test.df.iloc[i:iup],                             hditer_test.eta_range,                             hditer_test.phi_range,                             hditer_test.eta_bins,                             hditer_test.phi_bins)
    for c in range(3):
        x[:,c,:,:]/=hditer_train.max_abs[c]
    
    #write file
    f = h5.File(datadir+'/hep_test_chunk'+str(idx)+'.hdf5','w')
    f['data']=x
    f['label']=y
    f['weight']=w
    f['normweight']=w/hditer_train.wmax
    f['psr']=p
    f['mg']=mg
    f['mn']=mn
    f['jz']=jz
    f['eid']=eid
    f.close()


# ### Validation


print("Save validation files.")

for idx,i in tqdm(enumerate(range(0,hditer_validation.num_examples,chunksize_validation))):
    iup=np.min([i+chunksize_validation,hditer_validation.num_examples])
    
    #preprocess
    x,y,w,p,mg,mn,jz,eid = preprocess_data(hditer_validation.df.iloc[i:iup],                             hditer_validation.eta_range,                             hditer_validation.phi_range,                             hditer_validation.eta_bins,                             hditer_validation.phi_bins)
    for c in range(3):
        x[:,c,:,:]/=hditer_train.max_abs[c]
    
    #write the file
    f = h5.File(datadir+'/hep_validation_chunk'+str(idx)+'.hdf5','w')
    f['data']=x
    f['label']=y
    f['weight']=w
    #normalize those weights for validation to compare with training loss
    f['normweight']=w/hditer_train.wmax
    f['psr']=p
    f['mg']=mg
    f['mn']=mn
    f['jz']=jz
    f['eid']=eid
    f.close()


# # Test the dataset


testfiles=[x for x in os.listdir(datadir) if x.endswith('.hdf5')]



testresults=[]
for fname in tqdm(testfiles):
    f = h5.File(datadir+'/'+fname,'r')
    data=f['data'].value
    label=f['label'].value
    normweight=f['normweight'].value
    f.close()
    
    #scan filename for phase:
    phase=fname.split("_")[1]
    
    #compute stats:
    maxvals=np.max(data,axis=(2,3))
    minvals=np.min(data,axis=(2,3))
    normvals=np.linalg.norm(data,axis=(2,3))
    
    #compute density
    density=np.sum(data>0,axis=(2,3))/float(np.prod(data.shape[2:4]))
    
    #iterate over batches:
    for ind in range(data.shape[0]):
        tmpdict={'filename':fname, 'batchid': ind, 'phase':phase}
        
        tmpdict['max']=maxvals[ind,:]
        tmpdict['min']=minvals[ind,:]
        tmpdict['norm']=normvals[ind,:]
        tmpdict['label']=label[ind]
        tmpdict['normweight']=normweight[ind]
        tmpdict['density']=density[ind]
        testresults.append(tmpdict)

#store in dataframe
testdf=pd.DataFrame(testresults)


# ## Plot the result


#plot the results
nbins=100
nrows=3
numcols=3
fig, axvec= plt.subplots(figsize=(20*numcols, 10*nrows), nrows=nrows, ncols=numcols)

#use those colors
colors=['crimson','dodgerblue','aquamarine']

for row,typ in enumerate(['max','norm','density']):
    for channel in range(3):
        #get axis
        ax=axvec[row][channel]
        
        #set properties
        ax.set_title("Type: "+typ+" channel "+str(channel),fontproperties=font_prop_title)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(font_prop_axis)
            label.set_fontsize(32)
        #ax.set_xlabel('memory-mode',fontproperties=font_prop_axis)
        ax.set_ylabel('counts',fontproperties=font_prop_axis)
        
        #create histogram
        for idx,phase in enumerate(testdf.phase.unique()):
            #project data
            data=testdf[typ].ix[ testdf["phase"]==phase ].apply(lambda x: x[channel])
            Y, X = np.histogram(data, nbins, density=True)
            X=[ (X[i]+X[i+1])*0.5 for i in range(len(X)-1)]
            width=(X[1]-X[0])/3.
            ax.bar(X,Y,width=width, color=colors[idx])
        
plt.tight_layout()





