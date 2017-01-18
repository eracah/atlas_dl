
import matplotlib; matplotlib.use("agg")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_hdf('/global/project/projectdirs/das/wbhimji/RPVSusyJetLearn//DAOD_EXOT3.08548071._000001.pool.root.1.h5','caloclusters')



plt.figure(figsize=(30,30))
plt.imshow(np.log10(df['histo'][33]),interpolation='none')

plt.colorbar()



df_bg = pd.read_hdf('jj_DAOD_EXOT3.h5','caloclusters')



plt.imshow(df_bg['histo'][5],interpolation='none')
plt.colorbar()



fig, axes = plt.subplots(10, 10, figsize=(10, 10),
                         subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=0.05, wspace=0.05)
for i,ax in enumerate(axes.flat):
    im = ax.imshow(df['histo'][i],interpolation='none')
fig.colorbar(im, ax=axes.ravel().tolist())



df_bg





