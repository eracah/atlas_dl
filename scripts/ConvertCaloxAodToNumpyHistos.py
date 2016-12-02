import ROOT
import root_numpy as rnp
import pandas as pd
import numpy as np 
import sys 

inputFiles = sys.argv[1]

array = rnp.root2array(inputFiles, treename='CollectionTree',
                       branches=['CaloCalTopoClustersAuxDyn.calPhi',
                                 'CaloCalTopoClustersAuxDyn.calEta',
                                 'CaloCalTopoClustersAuxDyn.calE'],
                       start=0, stop=10000)

df = pd.DataFrame.from_records(array)

df['histo'] = map(lambda phi, eta, E : np.histogram2d(phi, eta, bins=100, weights=E,
                                                      range=[[-3.14, 3.14], [0., 2.]])[0],
                  df['CaloCalTopoClustersAuxDyn.calPhi'],
                  df['CaloCalTopoClustersAuxDyn.calEta'],
                  df['CaloCalTopoClustersAuxDyn.calE'])

df.to_hdf('data.h5','caloclusters')
