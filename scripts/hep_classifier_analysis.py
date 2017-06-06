
import matplotlib; matplotlib.use("agg")


__author__ = 'tkurth'
import sys
import os
import numpy as np
import scipy as sp
import pandas as pd
import h5py
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import time
from tqdm import tqdm
import re
import pickle


# ## Load Data


prefix='output'

#retrieve files
directory="/global/cscratch1/sd/tkurth/atlas_dl/atlas_caffe/hero_eval"
filelist=[x for x in os.listdir(directory) if x.startswith(prefix+'_chunk')]



filelist


# ## Populate Arrays


datalist=[]

for fname in tqdm(filelist):
    
    #determine chunkid:
    chunkid=int(re.match(prefix+r'_chunk(.*?).h5',fname).groups()[0])
    
    #print "Open file "+fname
    f = h5py.File(directory+'/'+fname,'r')
    
    tmpdata={}
    for item in f.items():
        
        #determine name and ID of item
        itemname=item[0].split('_')[0]
        itemid=int(item[0].split('_')[1])
        
        #read data
        data=list(f[item[0]].value)
        
        #add to dictionary
        if (chunkid,itemid) not in tmpdata.keys():
            tmpdata[(chunkid,itemid)]={itemname: data}
        else:
            tmpdata[(chunkid,itemid)][itemname]=data
    
    #close the file
    f.close()
    
    #put in list:
    for item in tmpdata:
        dct=tmpdata[item].copy()
        dct['chunk_id']=item[0]
        dct['item_id']=item[1]
        datalist.append(dct)



#Convert to dataframe:
dflist=[]
for item in tqdm(datalist):
    datareclist=[]
    for idx in range(len(item['softmax'])):
        tmpdict={'signal_prob': item['softmax'][idx][1],
                 'label': item['label'][idx],
                 'weight': item['weight'][idx],
                 'psr': item['psr'][idx],
                 'jz': item['jz'][idx],
                 'mg': item['mg'][idx],
                 'mn': item['mn'][idx]
                }
        datareclist.append(tmpdict)
    dflist.append(pd.DataFrame(datareclist))

#convert to dataframe
datadf=pd.concat(dflist)


# ## ROC


#seldf=datadf.ix[ ((datadf.mg==1400) & (datadf.mg==850)) | (datadf.jz!=0) ]
seldf=datadf



#compute ROC from data
fpr, tpr, thresholds = metrics.roc_curve(datadf.label, datadf.signal_prob, pos_label=1, sample_weight=datadf.weight)
fpr_cut, tpr_cut, thresholds = metrics.roc_curve(datadf.label, datadf.psr, pos_label=1, sample_weight=datadf.weight)
fpr_cut=fpr_cut[1]
tpr_cut=tpr_cut[1]



#plot the data
plt.figure()
lw = 2
#full curve
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr,tpr,reorder=True))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.scatter([fpr_cut],[tpr_cut], color='dodgerblue', label='standard cuts')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('plots/ROC_1400_850.png',dpi=300)
#pass

#zoomed-in
#plt.plot(fpr, tpr, color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr,tpr))
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 0.0004])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="upper right")
plt.savefig('plots/ROC_1400_850_zoom.png',dpi=300)
pass



#directly compare at the PSR FPR:
eff=np.interp([fpr_cut],fpr,tpr,0.1)[0]
eff_ratio=eff/tpr_cut
print("Efficiency: ",eff*100,"%")
print("Improvement in efficiency: ",eff_ratio)
print("Benchmark false positive rate:",fpr_cut)
print("Benchmark true positive rate:",tpr_cut)


# # CNN results


#threshold
threshold=0.999



#compute signal efficiency:
total_signal=np.sum(l_data)
found_signal=np.sum([1. if x[0]*x[1]>threshold else 0. for x in zip(l_data,p_data)])
print 'sig-efficiency(CNN): ',found_signal/total_signal
print 'sig-survivors(CNN): ',found_signal

#compute bg rejection:
total_background=np.sum([1.-x for x in l_data])
false_positive_bg=np.sum([1. if (1.-x[0])*x[1]>threshold else 0. for x in zip(l_data,p_data)])
print 'bg-rejection(CNN): ',1.-false_positive_bg/total_background
print 'bg-survivors(CNN): ',(1.-(1.-false_positive_bg/total_background))*total_background



#AMS
#compute s:
sval=np.sum([x[2] if x[0]*x[1]>threshold else 0. for x in zip(l_data,p_data,w_data)])
bval=np.sum([x[2] if (1.-x[0])*x[1]>threshold else 0. for x in zip(l_data,p_data,w_data)])
brval=10.

#print AMS results
print "AMS(CNN) = ",np.sqrt(2.*((sval+bval+brval)*np.log(1.+sval/(bval+brval))-sval))


# # Cut-based results


#compute signal efficiency:
total_signal=np.sum(l_data)
found_signal=np.sum([1. if x[0]*x[1]>0. else 0. for x in zip(l_data,c_data)])
print 'sig-efficiency(CUT): ',found_signal/total_signal
print 'sig-survivors(CUT): ',found_signal

#compute bg rejection:
total_background=np.sum([1.-x for x in l_data])
false_positive_bg=np.sum([1. if (1.-x[0])*x[1]>0. else 0. for x in zip(l_data,c_data)])
print 'bg-rejection(CUT): ',1.-false_positive_bg/total_background
print 'bg-survivors(CUT): ',(1.-(1.-false_positive_bg/total_background))*total_background



#AMS
#compute s:
sval=np.sum([x[2] if x[0]*x[1]>threshold else 0. for x in zip(l_data,c_data,w_data)])
bval=np.sum([x[2] if (1.-x[0])*x[1]>threshold else 0. for x in zip(l_data,c_data,w_data)])
brval=10.

#print AMS results
print "AMS(CUT) = ",np.sqrt(2.*((sval+bval+brval)*np.log(1.+sval/(bval+brval))-sval))





