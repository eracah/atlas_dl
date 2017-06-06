
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


eta_range = [-5,5]
eta_bins = 224
phi_range = [-3.1416, 3.1416]
phi_bins = 224



#inputfile="/global/cscratch1/sd/tkurth/atlas_dl/metadata/output_split/train_metadata_chunk0.db"
#dataoutputdir="/global/cscratch1/sd/tkurth/atlas_dl/data_delphes_final"
#argument parsing
parser = argparse.ArgumentParser(description='Preprocess Files for Training.')
parser.add_argument('--input', type=str, nargs=1, help='sqlite db which contains the data to be processed')
parser.add_argument('--output', type=str, nargs=1, help='hdf5 filename for final output')
args = parser.parse_args()

#fileparameters
inputfile=args.input[0]
outputfile=args.output[0]


# # Load metadatafile


con = sql.connect(inputfile)
metadatadf=pd.DataFrame(pd.read_sql("SELECT * FROM metadata;", con))
con.close()
#delete some unnecessary columns
del metadatadf["index"]
metadatadf["file"]=metadatadf["directory"]+"/"+metadatadf["filename"]
del metadatadf["directory"]
del metadatadf["filename"]
fgroups=metadatadf.groupby("file")


# # Process data


#prepare numpy arrays for IO
nsamples=metadatadf.shape[0]
x=np.zeros((nsamples,3,phi_bins,eta_bins),dtype=np.float32)
y=np.zeros((nsamples),dtype=np.int32)
w=np.zeros((nsamples),dtype=np.float32)
nw=np.zeros((nsamples),dtype=np.float32)
p=np.zeros((nsamples),dtype=np.int32)
mg=np.zeros((nsamples),dtype=np.float32)
mn=np.zeros((nsamples),dtype=np.float32)
jz=np.zeros((nsamples),dtype=np.int32)
eid=np.zeros((nsamples),dtype=np.int32)

#iterate over files and write to hdf5-file
for group in fgroups:
    gfname=group[0]
    gdf=group[1]
    
    #open file
    f = h5.File(gfname,'r')
    for row in gdf.iterrows():
        event=f[row[1]["id"]]
        arrid=row[0]
        
        #load data
        #channel-0
        clusPhi=event['clusPhi'].value
        clusEta=event['clusEta'].value
        clusE=event['clusE'].value
        
        
        #bin:
        x[arrid,0,:,:]=np.histogram2d(clusPhi,clusEta,
                                        bins=(phi_bins, eta_bins),weights=clusE,
                                        range=[phi_range,eta_range])[0]
        x[arrid,0,:,:]*=row[1]["clusE_norm"]
        
        #channel-1
        clusEM=event['clusEM'].value
        #bin:
        x[arrid,1,:,:]=np.histogram2d(clusPhi,clusEta,
                                        bins=(phi_bins, eta_bins),weights=clusEM,
                                        range=[phi_range,eta_range])[0]
        x[arrid,1,:,:]*=row[1]["clusEM_norm"]
        
        #channel-2
        trackPhi=event['trackPhi'].value
        trackEta=event['trackEta'].value
        #bin
        x[arrid,2,:,:]=np.histogram2d(trackPhi,trackEta,
                                        bins=(phi_bins, eta_bins),
                                        range=[phi_range,eta_range])[0]
        x[arrid,2,:,:]*=row[1]["track_norm"]
        
        #weights
        w[arrid]=event['weight'].value*row[1]["weight_norm"]
        nw[arrid]=event['weight'].value*row[1]["normweight_norm"]
        
        #label
        y[arrid]=row[1]["label"]
        
        #psr
        p[arrid]=int(event["passSR"].value)
        
        #mGlue
        mg[arrid]=row[1]["mGlu"]
        
        #mNeu
        mn[arrid]=row[1]["mNeu"]
        
        #JZ
        jz[arrid]=row[1]["jz"]
        
        #eid
        eid[arrid]=int(row[1]["id"].split("event_")[1])

    f.close()

#write out
f = h5.File(outputfile,'w')
f['data']=x
f['label']=y
f['weight']=w
f['normweight']=nw
f['psr']=p
f['mg']=mg
f['mn']=mn
f['jz']=jz
f['eid']=eid
f.close()





