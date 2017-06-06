
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


# # Useful Functions


#dict merger
def merge_dicts(dict1,dict2):
    tmp = dict1.copy()
    tmp.update(dict2)
    return tmp

#file string parser
def parse_filename(fname,directory='.'):
    directory=re.sub(r'^(.*?)(/+)$',r'\1',directory)
    
    #signal file?
    smatch=re.compile(r'^GG_RPV(.*?)_(.*?)_(.*?)_.*\.h5')
    tmpres=smatch.findall(fname)
    if tmpres:
        tmpres=tmpres[0]
        return {'rpv':int(tmpres[0]), 
                'mGlu':int(tmpres[1]), 
                'mNeu':int(tmpres[2]), 
                'jz': 0,
                'directory': directory,
                'filename': fname}

    #background file?
    smatch=re.compile(r'^jetjet_JZ(.*?)_.*\.h5')
    tmpres=smatch.findall(fname)
    if tmpres:
        return {'rpv': 0., 
                'mGlu': 0.,
                'mNeu': 0., 
                'jz': int(tmpres[0]),
                'directory': directory,
                'filename': fname}

    #nothing at all
    return {}


# # Global parameters


#read files from here
inputfile='/global/cscratch1/sd/tkurth/atlas_dl/metadata/inputfile.txt'
#directories to read from
#directories=['/global/cscratch1/sd/wbhimji/delphes_005_2017_03_06_NoPU-2',
#             '/global/cscratch1/sd/wbhimji/delphes_005_2017_03_06_NoPU']
#binning options
eta_range = [-5,5]
eta_bins = 224
phi_range = [-3.1416, 3.1416]
phi_bins = 224



#argument parsing
parser = argparse.ArgumentParser(description='Preprocess Files for Training.')
parser.add_argument('--input', type=str, nargs=1, help='file which contains list of input files')
parser.add_argument('--output', type=str, nargs=1, help='sqlite db file for storing metadata')
args = parser.parse_args()

#fileparameters
inputfile=args.input[0]
outputdbfilename=args.output[0]


# # Curate File List


#read input file:
filelist=[]
with open(inputfile) as f:
    lines=f.readlines()
    for line in lines:
        filelist.append(line)
    f.close()

#get mapping of files to directories:
filemap={}
for item in filelist:
    directory='/'.join(item.split('/')[:-1])
    filename=item.split('/')[-1].replace('\n','')
    if directory not in filemap:
        filemap[directory]=[filename]
    else:
        filemap[directory].append(filename)



print "Compiling Filelist"
filelist=[]
normlist=[]
for directory in filemap.keys():
    #load files
    filelist+=[parse_filename(x,directory) for x in filemap[directory]]
    #load normalizations:
    tmpdf=pd.read_csv(directory+'/DelphesNevents',sep=' ',index_col=False, header=None)
    tmpdf['directory']=directory
    normlist.append(tmpdf)
filedf=pd.DataFrame(filelist)
normdf=pd.concat(normlist)

#parse the normalizations:
normdf.rename(columns={1:'count'},inplace=True)
normdf[0]=normdf[0].str.replace('QCDBkg_','')
normdf['jz']=normdf[0].apply(lambda x: int(x.split('_')[0].split('JZ')[1]) if x.startswith('JZ') else 0)
normdf['rpv']=normdf[0].apply(lambda x: int(x.split('_')[1].split('RPV')[1]) if x.startswith('GG') else 0.)
normdf['mGlu']=normdf[0].apply(lambda x: int(x.split('_')[2]) if x.startswith('GG') else 0.)
normdf['mNeu']=normdf[0].apply(lambda x: int(x.split('_')[3]) if x.startswith('GG') else 0.)

#merge with filedf
filedf=filedf.merge(normdf[['count','directory','rpv','jz','mGlu','mNeu']],how='left',on=['directory','rpv','jz','mGlu','mNeu'])

#sort
filedf.sort_values(by=['directory','filename'],inplace=True)
filedf.reset_index(drop=True,inplace=True)


# ## Preprocess Data


#create connection to sql-db:
print "Establishing DB connection"
con = sql.connect(outputdbfilename)



#iterate over files, compute the max value for given binning and for weights:
print "Processing Filelist"
datadflist=[]
for row in filedf.iterrows():
    #open file
    f = h5.File(row[1]['directory']+'/'+row[1]['filename'],'r')
    
    #iterate over items
    rowlist=[]
    for item in f.iteritems():
        if not item[0].startswith('event'):
            continue
        
        #copy rowdict
        tmpdict=row[1].copy()
        
        #name
        tmpdict["id"]=item[0]
        
        #channel-0
        clusPhi=item[1]['clusPhi'].value
        clusEta=item[1]['clusEta'].value
        clusE=item[1]['clusE'].value
        #bin:
        tmpdict["clusE_max"]=np.max(np.histogram2d(clusPhi,clusEta,
                                    bins=(phi_bins, eta_bins),weights=clusE,
                                    range=[phi_range,eta_range])[0])
        
        #channel-1
        clusEM=item[1]['clusEM'].value
        #bin:
        tmpdict["clusEM_max"]=np.max(np.histogram2d(clusPhi,clusEta,
                                    bins=(phi_bins, eta_bins),weights=clusEM,
                                    range=[phi_range,eta_range])[0])
        
        #channel-2
        trackPhi=item[1]['trackPhi'].value
        trackEta=item[1]['trackEta'].value
        #bin
        tmpdict["track_max"]=np.max(np.histogram2d(trackPhi,trackEta,
                                    bins=(phi_bins, eta_bins),
                                    range=[phi_range,eta_range])[0])
        
        #weight
        tmpdict["weight_max"]=np.max(item[1]['weight'].value)
        
        #append to list of rows
        rowlist.append(tmpdict)
        
    #close the file
    f.close()
    
    #write to database:
    pd.DataFrame(rowlist).to_sql("metadata", con, if_exists='append',chunksize=200)

#close connection
con.close()



print "Done"

