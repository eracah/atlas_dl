
import matplotlib; matplotlib.use("agg")


__author__ = 'tkurth'
import sys
import os
import numpy as np
from scipy.stats import cumfreq
import pandas as pd
import h5py
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import time
import re
from tqdm import tqdm
import pickle


# # Useful Functions


def calc_cdf(data, weights=None, normalize=True):
    
    #sort the values first
    if weights:
        data=np.vstack([data,weights])
    else:
        data=np.vstack([data,np.ones(len(data))])
    data=np.sort(data,axis=1)
    
    #normalize if requested
    if normalize:
        data[1][:]=data[1][:]/np.sum(data[1][:])
    
    #integrate
    X=data[0][:]
    Y=[data[1][0]]    
    for i in range(1,data.shape[1]):
        Y.append(Y[i-1]+data[1][i])
    Y=np.asarray(Y)
    
    #return result
    return X,Y


#compute quantiles:
def get_quantile(cdf, ptarget):

    #invert CDF
    idx=0
    pcurrent=cdf[1][idx]
    while (pcurrent<ptarget) and (idx<len(cdf[1][:])):
        idx+=1
        pcurrent=cdf[1][idx]

    #report x when done
    return cdf[0][idx]



# ## Load Fitting Data


#retrieve files
directory="/global/cscratch1/sd/tkurth/atlas_dl/atlas_caffe/autoencoder/runs_preselect_autoencoder"
filelist=[x for x in os.listdir(directory) if x.startswith('output_fit_chunk')]


# ## Populate Fitting Arrays


fitdatalist=[]
for fname in tqdm(filelist):
    
    #determine chunkid:
    chunkid=int(re.match(r'^.*?chunk(.*?).h5',fname).groups()[0])
    
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
        fitdatalist.append(dct) 



#Convert to stacked array:
p_data=[]
w_data=[]
for item in fitdatalist:
    p_data+=item['loss']
    w_data+=item['weight']


# ## Plot CDF


X,Y=calc_cdf(p_data,w_data)
plt.plot(X,Y)



#compute quantiles:
pvalue_target=0.999

#report x when done
x_target=get_quantile(np.vstack([X,Y]),pvalue_target)

#print the value
print "Exclusion threshold: ",x_target


# # Anomaly Detection

# ## Load Test Data


#retrieve files
directory="/global/cscratch1/sd/tkurth/atlas_dl/atlas_caffe/autoencoder/runs_preselect_autoencoder"
filelist=[x for x in os.listdir(directory) if x.startswith('output_chunk')]


# ## Populate Arrays


datalist=[]

for fname in tqdm(filelist):
    
    #determine chunkid:
    chunkid=int(re.match(r'output_chunk(.*?).h5',fname).groups()[0])
    
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



#unfold datalist
cleandatalist=[]
for idx,item in enumerate(datalist):
    p_data=item['loss']
    w_data=item['weight']
    mg_data=item['mg']
    mn_data=item['mn']
    psr_data=item['psr']
    
    for idy in range(len(p_data)):
        if psr_data[idy]>0.:
            psr_val=True
        else:
            psr_val=False
        
        cleandatalist.append(
                            {'loss': p_data[idy],
                             'weight': w_data[idy],
                             'mg': mg_data[idy],
                             'mn': mn_data[idy],
                             'psr': psr_val
                            }
                            )

#convert to stacked arrays:
datadf=pd.DataFrame(cleandatalist)
datadf.sort_values(by=['mg','mn'],inplace=True)
datadf.reset_index(drop=True,inplace=True)

#split in background and signal sets
bgdf=datadf.ix[ (datadf.mg==0.) & (datadf.mn==0.) ].copy()
sigdf=datadf.ix[ (datadf.mg>0.) | (datadf.mn>0.) ].copy()


# ## Results


#threshold
threshold=0.01

#compute background rejection
bval_per=bgdf['weight'].ix[ bgdf.loss<threshold ].sum()
bval_psr=bgdf['weight'].ix[ bgdf.psr==True ].sum()

#compute signal efficiencies
#PER
resultdf=pd.DataFrame(sigdf.groupby(['mg','mn']).apply(lambda x: x['weight'].ix[ x.loss<threshold ].sum()))
resultdf.reset_index(inplace=True)
resultdf.rename(columns={0:'sval_per'},inplace=True)
#PSR
tmpdf=pd.DataFrame(sigdf.groupby(['mg','mn']).apply(lambda x: x['weight'].ix[ x.psr==True ].sum()))
tmpdf.reset_index(inplace=True)
tmpdf.rename(columns={0:'sval_psr'},inplace=True)

#merge the two
resultdf=resultdf.merge(tmpdf,on=['mg','mn'], how='left')

#compute AMS
brval=10.

#print AMS results
resultdf["ams_per"]=np.sqrt(2.*((resultdf.sval_per+bval_per+brval)*np.log(1.+resultdf.sval_per/(bval_per+brval))-resultdf.sval_per))
resultdf["ams_psr"]=np.sqrt(2.*((resultdf.sval_psr+bval_psr+brval)*np.log(1.+resultdf.sval_psr/(bval_psr+brval))-resultdf.sval_psr))



resultdf


# ## ROC


#compute ROC from data
fpr, tpr, thresholds = metrics.roc_curve(l_data, p_data, pos_label=1, sample_weight=w_data)
fpr_cut, tpr_cut, thresholds = metrics.roc_curve(l_data, c_data, pos_label=1, sample_weight=w_data)
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

#zoomed-in
#plt.plot(fpr, tpr, color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr,tpr))
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 0.0002])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="upper right")
plt.savefig('plots/ROC_1400_850_zoom.png',dpi=300)
pass


# # CNN results


#set threshold
threshold=0.5

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
threshold=0.8

#compute s:
sval=np.sum([x[2] if x[0]*x[1]>threshold else 0. for x in zip(l_data,p_data,w_data)])
bval=np.sum([x[2] if (1.-x[0])*x[1]>threshold else 0. for x in zip(l_data,p_data,w_data)])
brval=10.

#print AMS results
print "AMS(CNN) = ",np.sqrt(2.*((sval+bval+brval)*np.log(1.+sval/(bval+brval))-sval))


# # Cut-based results


#set threshold
threshold=0.9

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
threshold=0.8

#compute s:
sval=np.sum([x[2] if x[0]*x[1]>threshold else 0. for x in zip(l_data,c_data,w_data)])
bval=np.sum([x[2] if (1.-x[0])*x[1]>threshold else 0. for x in zip(l_data,c_data,w_data)])
brval=10.

#print AMS results
print "AMS(CUT) = ",np.sqrt(2.*((sval+bval+brval)*np.log(1.+sval/(bval+brval))-sval))





