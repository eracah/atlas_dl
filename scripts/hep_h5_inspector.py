
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
import h5py as h5




datadir='/global/cscratch1/sd/tkurth/atlas_dl/data_preselect_augmented'
trainfiles=[x for x in os.listdir(datadir) if x.endswith('hdf5') and x.startswith('hep_training') ]



traindata=[]
for fname in trainfiles:
    #open the file
    f = h5.File(datadir+'/'+fname,'r')
    l=f['label'].value
    w=f['normweight'].value
    f.close()
    
    reslist=[{'file':fname,'entry':idx,'label':x[0],'weight':x[1]} for idx,x in enumerate(zip(l,w))]
    traindata+=reslist

alldf=pd.DataFrame(traindata)



Ybg,Xbg = np.histogram(alldf['weight'].ix[ alldf.label==0], bins=10)
Ysig,_ = np.histogram(alldf['weight'].ix[ alldf.label==1], bins=Xbg)

#average X:
X=[(Xbg[i]+Xbg[i+1])*0.5 for i in range(len(Xbg)-1)]
width=(Xbg[1]-Xbg[0])*0.25

#plot histogram
lw=2
plt.bar(X-width,Ybg, width=width, color='darkorange', lw=lw, label='Background Weights')
plt.bar(X+width,Ysig, width=width, color='dodgerblue', lw=lw, label='Signal Weights')
plt.yscale('log')
plt.xlabel('weight')
plt.ylabel('#entries')
plt.legend(loc="upper right")
plt.savefig('plots/learning_curve.png',dpi=300)



alldf.sort_values('weight',ascending=False)





