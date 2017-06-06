
import matplotlib; matplotlib.use("agg")


__author__ = 'tkurth'
import sys
import os
import argparse
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
import h5py as h5

#sqlite for storing the metadata
import sqlite3 as sql


# # Global Parameters


metadatadirs=["/global/cscratch1/sd/tkurth/atlas_dl/metadata/64x64/create_metadata/output"]
metadataoutputdir="/global/cscratch1/sd/tkurth/atlas_dl/metadata/64x64/create_metadata/output_split"
restart=False
numnodes=4096
#parameters
train_fraction=0.75
validation_fraction=0.10
nsig_augment=1
#binning options
eta_range = [-5,5]
eta_bins = 64
phi_range = [-3.1416, 3.1416]
phi_bins = 64
#jz cut:
jzmin=3
jzmax=11
#SUSY theory:
trainselect=[{'mGlu':1400, 'mNeu': 850}]


# # Load Metadata


metadatafiles=[]
for directory in metadatadirs:
    #load files
    metadatafiles=[directory+'/'+x for x in os.listdir(directory) if x.endswith('.db')]
    
#read the database-files
dflist=[]
for mf in metadatafiles:
    con = sql.connect(mf)
    tmpdf=pd.DataFrame(pd.read_sql("SELECT * FROM metadata;", con))
    con.close()
    
    #clean up
    del tmpdf['index']
    dflist.append(tmpdf)

#concatenate
datadf=pd.concat(dflist)



print "Size of dataframe in GB:",sys.getsizeof(datadf)/(1024*1024*1024)



print("Split by signal and BG.")

#select signal and background configuration
siglist=[]
for item in trainselect:
    siglist.append(datadf[ (datadf['mGlu']==item['mGlu']) & (datadf['mNeu']==item['mNeu']) ])
sigdf=pd.concat(siglist)

#select background configuration
bgdf=datadf[ (datadf['jz']>=jzmin) & (datadf['jz']<=jzmax) ]

print("Determine frequencies.")

#background:
bggroup=bgdf.groupby(['jz','directory'])
tmpdf=pd.DataFrame(bggroup['id'].count())
tmpdf.reset_index(inplace=True)
tmpdf.rename(columns={'id':'frequency'},inplace=True)
bgdf=bgdf.merge(tmpdf,on=['jz','directory'],how='left')

#signal:
siggroup=sigdf.groupby(['mGlu','mNeu','directory'])
tmpdf=pd.DataFrame(siggroup['id'].count())
tmpdf.reset_index(inplace=True)
tmpdf.rename(columns={'id':'frequency'},inplace=True)
sigdf=sigdf.merge(tmpdf,on=['mGlu','mNeu','directory'],how='left')



#do the splitting
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


print "We got the following total frequencies:"
print "Training set: #signal = ",trainsigdf.shape[0]," #background = ",trainbgdf.shape[0]
print "Validation set: #signal = ",valsigdf.shape[0]," #background = ",valbgdf.shape[0]
print "Test set: #signal = ",testsigdf.shape[0]," #background = ",testbgdf.shape[0]


# # Split into input files for multi-node processing


#combine the dataframes:
#train
traindf=pd.concat([trainsigdf,trainbgdf])
traindf.reset_index(inplace=True,drop=True)

#validation
validf=pd.concat([valsigdf,valbgdf])
validf.reset_index(inplace=True,drop=True)

#test
testdf=pd.concat([testsigdf,testbgdf])
testdf.reset_index(inplace=True,drop=True)


#shuffle:
#train
np.random.seed(13)
traindf=traindf.reindex(np.random.permutation(traindf.index))
traindf.reset_index(inplace=True,drop=True)

#validation
np.random.seed(13)
validf=validf.reindex(np.random.permutation(validf.index))
validf.reset_index(inplace=True,drop=True)

#test
np.random.seed(13)
testdf=testdf.reindex(np.random.permutation(testdf.index))
testdf.reset_index(inplace=True,drop=True)



#print ensemble sizes and determine the chunk size
chunksize_train=int(traindf.shape[0]/float(numnodes))
print "Training size: ",int(traindf.shape[0]),' chunk size: ',chunksize_train
chunksize_validation=int(validf.shape[0]/float(numnodes))
print "Validation size: ",validf.shape[0],' chunk size: ',chunksize_validation
chunksize_test=int(testdf.shape[0]/float(numnodes))
print "Test size: ",testdf.shape[0],' chunk size: ',chunksize_test


# # Store normalizations and throw away stuff which is not needed


#normalize weights
maxnormweight=np.max(traindf['weight_max']/traindf['count'])
#normweight
traindf["normweight_norm"]=1./(traindf['count']*maxnormweight)
validf["normweight_norm"]=1./(validf['count']*maxnormweight)
testdf["normweight_norm"]=1./(testdf['count']*maxnormweight)
#regular weight
traindf["weight_norm"]=1./traindf['count']
validf["weight_norm"]=1./validf['count']
testdf["weight_norm"]=1./testdf['count']
#normalize input channels
#clusE
max_clusE_max=np.max(traindf['clusE_max'])
traindf['clusE_norm']=1./max_clusE_max
validf['clusE_norm']=1./max_clusE_max
testdf['clusE_norm']=1./max_clusE_max
#clusEM
max_clusEM_max=np.max(traindf['clusEM_max'])
traindf['clusEM_norm']=1./max_clusEM_max
validf['clusEM_norm']=1./max_clusEM_max
testdf['clusEM_norm']=1./max_clusEM_max
#track
max_track_max=float(np.max(traindf['track_max']))
traindf['track_norm']=1./max_track_max
validf['track_norm']=1./max_track_max
testdf['track_norm']=1./max_track_max

#now, set label
#train
traindf['label']=0
traindf.loc[traindf.jz == 0., 'label']=1
#validation
validf['label']=0
validf.loc[validf.jz == 0., 'label']=1
#test
testdf['label']=0
testdf.loc[testdf.jz == 0., 'label']=1


# # Write out splitted dataframes so that individual processes can chew trough the data


#see which files are done
ind_done={'train':[], 'validation':[], 'test':[]}
if not restart:
    filelist=[x for x in os.listdir(metadataoutputdir) if x.endswith('.db')]
    for fname in filelist:
        phasename=fname.split('_')[0]
        ind=int(fname.split('_chunk')[1].split('.db')[0])
        ind_done[phasename].append(ind)



ind_done



#training
for ind in range(numnodes):
    #skip if already done:
    if ind in ind_done['train']:
        continue
    
    #upper and lower index
    lo=ind*chunksize_train
    up=(ind+1)*chunksize_train
    
    #get slice:
    seldf=traindf[['directory','filename','id','label','normweight_norm','weight_norm','clusE_norm','clusEM_norm','track_norm','mGlu','mNeu','jz']].iloc[lo:up,:]
    
    #establish db connection
    con = sql.connect(metadataoutputdir+'/train_metadata_chunk'+str(ind)+'.db')
    
    #write out:
    seldf.to_sql("metadata", con, if_exists='replace',chunksize=200)
    
    #close connection:
    con.close()

#validation
for ind in range(numnodes):
    #skip if already done:
    if ind in ind_done['validation']:
        continue
    
    #upper and lower index
    lo=ind*chunksize_validation
    up=(ind+1)*chunksize_validation
    
    #get slice:
    seldf=validf[['directory','filename','id','label','normweight_norm','weight_norm','clusE_norm','clusEM_norm','track_norm','mGlu','mNeu','jz']].iloc[lo:up,:]
    
    #establish db connection
    con = sql.connect(metadataoutputdir+'/validation_metadata_chunk'+str(ind)+'.db')
    
    #write out:
    seldf.to_sql("metadata", con, if_exists='replace',chunksize=200)
    
    #close connection:
    con.close()

#test
for ind in range(numnodes):
    #skip if already done:
    if ind in ind_done['test']:
        continue
    
    #upper and lower index
    lo=ind*chunksize_test
    up=(ind+1)*chunksize_test
    
    #get slice:
    seldf=testdf[['directory','filename','id','label','normweight_norm','weight_norm','clusE_norm','clusEM_norm','track_norm','mGlu','mNeu','jz']].iloc[lo:up,:]
    
    #establish db connection
    con = sql.connect(metadataoutputdir+'/test_metadata_chunk'+str(ind)+'.db')
    
    #write out:
    seldf.to_sql("metadata", con, if_exists='replace',chunksize=200)
    
    #close connection:
    con.close()





