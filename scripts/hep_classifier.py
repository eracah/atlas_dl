
import matplotlib; matplotlib.use("agg")

# ## General modules


__author__ = 'tkurth'
import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import time
import re
import pickle


# ## Theano modules


import theano
import theano.tensor as T
import theano.sandbox.rng_mrg
Trng = theano.sandbox.rng_mrg.MRG_RandomStreams(9)
import lasagne as ls


# ## ROOT stuff


sys.path.append('/global/homes/w/wbhimji/cori-envs/nersc-rootpy/lib/python2.7/site-packages/')
sys.path.append('/global/common/cori/software/root/6.06.06/lib/root')
import ROOT
import rootpy
import root_numpy as rnp


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


# ## Data loader and preprocessor


def merge_dicts(dict1,dict2):
    tmp = dict1.copy()
    tmp.update(dict2)
    return tmp



#file string parser
def parse_filename(fname,directory='.'):
    directory=re.sub(r'^(.*?)(/+)$',r'\1',directory)
    
    #signal file?
    smatch=re.compile(r'GG_RPV(.*?)_(.*?)_(.*?)\.merge')
    tmpres=smatch.findall(fname)
    if tmpres:
        tmpres=tmpres[0]
        return {'rpv':int(tmpres[0]), 'mass1':int(tmpres[1]), 'mass2':int(tmpres[2]), 'name':directory+'/'+fname}

    #background file?
    smatch=re.compile(r'JZ(.*?)\.merge')
    tmpres=smatch.findall(fname)
    if tmpres:
        return {'jz':int(tmpres[0]), 'name':directory+'/'+fname}

    #nothing at all
    return {}



def load_data(filelists,
                group_name='CollectionTree',
                branches=['CaloCalTopoClustersAuxDyn.calPhi', \
                          'CaloCalTopoClustersAuxDyn.calEta', \
                          'CaloCalTopoClustersAuxDyn.calE'],
                dataset_name='histo',
                type_='root'):
    
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
        
        #read the files in the filelist
        files = [line.rstrip() for line in open(fname)]
        
        #we don't want annoying stderr messages
        with suppress_stdout_stderr():
            
            #bgarray has n_events groups of 3 parallel numpy arrays 
            #(each numpy within a group is of equal length and each array corresponds to phi, eta and the corresponding energy)
            try:
                datarec = rnp.root2array(files,                                         treename=group_name,                                         branches=branches,                                         start=0,                                         warn_missing_tree=True)
                tmpdf=pd.DataFrame.from_records(datarec)
                reclist=tmpdf[['CaloCalTopoClustersAuxDyn.calPhi',                                 'CaloCalTopoClustersAuxDyn.calEta',                                 'CaloCalTopoClustersAuxDyn.calE']].to_dict('records')
                reclist=[merge_dicts(masterrec,rec) for rec in reclist]
                    
            except:
                continue
            
        #append to records
        records+=reclist
            
    #return dataframe
    return pd.DataFrame(records)


#preprocessor
def preprocess_data(df,eta_range,phi_range,eta_bins,phi_bins):
    #empty array
    xvals = np.zeros((df.shape[0], 1, phi_bins, eta_bins ),dtype='float32')
    yvals = np.zeros((df.shape[0],),dtype='int32')
    
    for i in range(df.shape[0]):        
        phi, eta, E =  df.iloc[i]['CaloCalTopoClustersAuxDyn.calPhi'],                       df.iloc[i]['CaloCalTopoClustersAuxDyn.calEta'],                       df.iloc[i]['CaloCalTopoClustersAuxDyn.calE']
        
        xvals[i]=np.histogram2d(phi,eta,
                                bins=(phi_bins, eta_bins), \
                                weights=E,
                                range=[phi_range,eta_range])[0]
        yvals[i]=df.iloc[i]['label']
        
    return xvals, yvals


# ## Data iterator


class hep_data_iterator:
    
    #class constructor
    def __init__(self,
                 datadf,
                 max_frequency=None,
                 even_frequencies=True,
                 shuffle=True,
                 nbins=(100,100),
                 eta_range = [-5,5],
                 phi_range = [-3.1416, 3.1416]
                ):

        #set parameters
        self.shuffle = shuffle
        self.nbins = nbins
        self.eta_range = eta_range
        self.phi_range = phi_range
        self.even_frequencies = even_frequencies
        
        #compute bins
        self.eta_bins = self.nbins[0]
        self.phi_bins = self.nbins[1]
        
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
        elif not even_frequencies:
            min_frequency=-1
        tmpdf=self.df.groupby(['label']).apply(lambda x: x[['CaloCalTopoClustersAuxDyn.calPhi',                                                             'CaloCalTopoClustersAuxDyn.calEta',                                                             'CaloCalTopoClustersAuxDyn.calE']].iloc[:min_frequency,:]).copy()
        tmpdf.reset_index(inplace=True)
        del tmpdf['level_1']
        
        #copy tmpdf into self.df
        self.df=tmpdf.copy()
        
        #compute max:
        self.compute_data_max()
        
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
    
    
    #this is the batch iterator:
    def next_batch(self,batchsize):
        '''batch iterator'''
        
        #shuffle:
        if self.shuffle:
            self.df=self.df.reindex(np.random.permutation(self.df.index))
        
        #iterate
        for idx in range(0,self.num_examples-batchsize,batchsize):
            #yield next batch
            x,y=preprocess_data(self.df.iloc[idx:idx+batchsize,:],                             self.eta_range,
                             self.phi_range,
                             self.eta_bins,self.phi_bins)
            #rescale x:
            x/=self.max_abs
        
            #return result
            yield x,y


# ## Curate File list


directory='/project/projectdirs/das/wbhimji/RPVSusyJetLearn/atlas_dl/config/'
filelists=[parse_filename(x,directory) for x in os.listdir(directory) if x.startswith('mc')]
filenamedf=pd.DataFrame(filelists)



#select signal configuration
mass1=1400
mass2=850
sig_cfg_files=list(filenamedf[ (filenamedf['mass1']==1400) & (filenamedf['mass2']==850) ]['name'])

#select background configuration
jzmin=4
jzmax=5
bg_cfg_files=list(filenamedf[ (filenamedf['jz']>=jzmin) & (filenamedf['jz']<=jzmax) ]['name'])


# ## Construct data iterator


#load background files
bgdf=load_data(bg_cfg_files)
bgdf=bgdf.reindex(np.random.permutation(bgdf.index))



#load signal data
sigdf=load_data(sig_cfg_files)
sigdf=sigdf.reindex(np.random.permutation(sigdf.index))



#parameters
train_fraction=0.8
validation_fraction=0.1
nbins=(100,100)

#create sizes:
num_sig_train=int(np.floor(sigdf.shape[0]*train_fraction))
#num_bg_train=int(np.floor(bgdf.shape[0]*train_fraction))
num_bg_train=num_sig_train
num_sig_validation=int(np.floor(sigdf.shape[0]*validation_fraction))
#num_bg_validation=int(np.floor(bgdf.shape[0]*validation_fraction))
num_bg_validation=num_sig_validation

#split the sets
traindf=pd.concat([bgdf.iloc[:num_bg_train],sigdf.iloc[:num_sig_train]])
validdf=pd.concat([bgdf.iloc[num_bg_train:num_bg_train+num_bg_validation],                    sigdf.iloc[num_sig_train:num_sig_train+num_sig_validation]])
testdf=pd.concat([bgdf.iloc[num_bg_train+num_bg_validation:],                    sigdf.iloc[num_sig_train+num_sig_validation:]])

#create iterators
hditer_train=hep_data_iterator(traindf,nbins=nbins)
hditer_validation=hep_data_iterator(validdf,nbins=nbins)
hditer_test=hep_data_iterator(testdf,nbins=nbins,even_frequencies=False)

#the preprocessing for the validation iterator has to be taken from the training iterator
hditer_validation.max_abs=hditer_train.max_abs
hditer_test.max_abs=hditer_train.max_abs



print hditer_train.num_examples
print hditer_validation.num_examples
print hditer_test.num_examples


# # Classifier


#Matthews correlation coefficient objective, only for binary classifications
def matthews_correlation_coefficient(predictions, targets):
    #preprocess
    if targets.ndim == predictions.ndim:
        targets = T.argmax(targets, axis=-1)
    #make predictions flat as well:
    predictions = T.argmax(predictions, axis=-1)
    
    #true predictions
    true_pred=T.eq(predictions, targets)
    #false predictions
    false_pred=T.neq(predictions, targets)
    
    #true positives:
    tp=(true_pred*predictions).sum()
    #false positives:
    fp=(false_pred*predictions).sum()
    #true negatives
    tn=(true_pred*(1-predictions)).sum()
    #false negatives
    fn=(false_pred*(1-predictions)).sum()
    
    #now, assemble ratio
    mcc=(tp*tn-fp*fn)/T.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    
    return mcc



#debug
ypred=T.matrix('ypred')
ytrue=T.imatrix('ytrue')
mccdummy=matthews_correlation_coefficient(ypred,ytrue)
mcc=theano.function([ypred,ytrue],mccdummy)



mcc(np.asarray([[0.3,0.7],[0.8,0.2],[0.56,0.44],[0.1,0.9]],dtype=np.float32),np.asarray([[0,1],[1,0],[1,0],[0,1]],dtype=np.int32))


# ## Construct classification network


#some parameters
keep_prob=0.5
num_filters=128
num_units_dense=1024
initial_learning_rate=0.001

#input layer
l_inp_data = ls.layers.InputLayer((None,hditer_train.xshape[0],hditer_train.xshape[1],hditer_train.xshape[2]))
l_inp_label = ls.layers.InputLayer((None,1))

#conv layers
#first layer
l_conv1 = ls.layers.Conv2DLayer(incoming=l_inp_data,
                                num_filters=num_filters,
                                filter_size=3,
                                stride=(1,1),
                                pad=0,
                                W=ls.init.HeUniform(),
                                b=ls.init.Constant(0.),
                                nonlinearity=ls.nonlinearities.LeakyRectify()
                               )
l_drop1 = ls.layers.DropoutLayer(incoming=l_conv1,
                       p=keep_prob,
                       rescale=True
                      )
l_pool1 = ls.layers.MaxPool2DLayer(incoming=l_drop1,
                                   pool_size=(2,2),
                                   stride=2,
                                   pad=0                                   
                                  )

#second layer:
l_conv2 = ls.layers.Conv2DLayer(incoming=l_pool1,
                                num_filters=num_filters,
                                filter_size=3,
                                stride=(1,1),
                                pad=0,
                                W=ls.init.HeUniform(),
                                b=ls.init.Constant(0.),
                                nonlinearity=ls.nonlinearities.LeakyRectify()
                               )
l_drop2 = ls.layers.DropoutLayer(incoming=l_conv2,
                       p=keep_prob,
                       rescale=True
                      )
l_pool2 = ls.layers.MaxPool2DLayer(incoming=l_drop2,
                                   pool_size=(2,2),
                                   stride=2,
                                   pad=0                                   
                                  )

#third layer:
l_conv3 = ls.layers.Conv2DLayer(incoming=l_pool2,
                                num_filters=num_filters,
                                filter_size=3,
                                stride=(1,1),
                                pad=0,
                                W=ls.init.HeUniform(),
                                b=ls.init.Constant(0.),
                                nonlinearity=ls.nonlinearities.LeakyRectify()
                               )
l_drop3 = ls.layers.DropoutLayer(incoming=l_conv3,
                       p=keep_prob,
                       rescale=True
                      )
l_pool3 = ls.layers.MaxPool2DLayer(incoming=l_drop3,
                                   pool_size=(2,2),
                                   stride=2,
                                   pad=0                                   
                                  )

#fourth layer:
l_conv4 = ls.layers.Conv2DLayer(incoming=l_pool3,
                                num_filters=num_filters,
                                filter_size=3,
                                stride=(1,1),
                                pad=0,
                                W=ls.init.HeUniform(),
                                b=ls.init.Constant(0.),
                                nonlinearity=ls.nonlinearities.LeakyRectify()
                               )
l_drop4 = ls.layers.DropoutLayer(incoming=l_conv4,
                       p=keep_prob,
                       rescale=True
                      )
l_pool4 = ls.layers.MaxPool2DLayer(incoming=l_drop4,
                                   pool_size=(2,2),
                                   stride=2,
                                   pad=0                                   
                                  )

#flatten
l_flat = ls.layers.FlattenLayer(incoming=l_pool4, 
                                outdim=2)

#crossfire
l_fc1 = ls.layers.DenseLayer(incoming=l_flat, 
                             num_units=num_units_dense, 
                             W=ls.init.GlorotUniform(np.sqrt(2./(1+0.01**2))), 
                             b=ls.init.Constant(0.0),
                             nonlinearity=ls.nonlinearities.LeakyRectify()
                            )

l_drop5 = ls.layers.DropoutLayer(incoming=l_fc1,
                       p=keep_prob,
                       rescale=True
                      )

l_fc2 = ls.layers.DenseLayer(incoming=l_drop5, 
                             num_units=num_units_dense, 
                             W=ls.init.GlorotUniform(np.sqrt(2./(1+0.01**2))), 
                             b=ls.init.Constant(0.0),
                             nonlinearity=ls.nonlinearities.LeakyRectify()
                            )

l_drop6 = ls.layers.DropoutLayer(incoming=l_fc2,
                       p=keep_prob,
                       rescale=True
                      )

#output layer
l_out = ls.layers.DenseLayer(incoming=l_drop6, 
                             num_units=hditer_train.num_classes, 
                             W=ls.init.GlorotUniform(np.sqrt(2./(1+0.01**2))), 
                             b=ls.init.Constant(0.0),
                             nonlinearity=ls.nonlinearities.softmax
                            )

#network
network = [l_inp_data, l_inp_label,
           l_conv1, l_pool1, l_drop1,
           l_conv2, l_pool2, l_drop2,
           l_conv3, l_pool3, l_drop3,
           l_conv4, l_pool4, l_drop4,
           l_flat, 
           l_fc1, l_drop5,
           l_fc2, l_drop6,
           l_out
          ]

#variables
inp = l_inp_data.input_var
lab = T.ivector('lab')

#output
lab_pred = ls.layers.get_output(l_out, {l_inp_data: inp})
lab_pred_det = ls.layers.get_output(l_out, {l_inp_data: inp}, deterministic=True)

#loss functions:
loss = ls.objectives.categorical_crossentropy(lab_pred,lab).mean()
loss_det = ls.objectives.categorical_crossentropy(lab_pred_det,lab).mean()

#accuracy
acc_det = ls.objectives.categorical_accuracy(lab_pred_det, lab, top_k=1).mean()

#MCC
mcc_det = matthews_correlation_coefficient(lab_pred_det,lab)

#parameters
params = ls.layers.get_all_params(network, trainable=True)

#updates
updates = ls.updates.adam(loss, params, learning_rate=initial_learning_rate)

#compile network function
fnn = theano.function([inp], lab_pred)
fnn_det = theano.function([inp], lab_pred_det)
#training function to minimize
fnn_train = theano.function([inp,lab], loss, updates=updates)
#validation function with accuracy
fnn_validate = theano.function([inp,lab], [loss_det,acc_det,mcc_det])


# ## Load Network


#load from which files
load_model=False
paramsfile_load="model_parameters.pick"
updatesfile_load="model_updates.pick"



if load_model:
    #load updates
    with open(updatesfile_load,'rb') as f:
        values1=pickle.load(f)
        f.close()
        for p, value in zip(updates.keys(), values1):
            p.set_value(value)
    
    #load parameters
    with open(paramsfile_load,'rb') as f:
        values2=pickle.load(f)
        f.close()
        ls.layers.set_all_param_values(network, values2, trainable=True)


# ## Train classifier


train_model=True
num_epochs=10
batchsize=128
paramsfile_savebest="model_parameters_best.pick"
updatesfile_savebest="model_updates_best.pick"



if train_model:
    
    #validation error
    best_val_err=1.e6
    
    for epoch in range(num_epochs):
        train_err = 0.
        train_acc = 0.
        train_mcc = 0.
        train_batches = 0.
        start_time = time.time()
        for batch in hditer_train.next_batch(batchsize):
            inputs, targets = batch
            train_err += fnn_train(inputs, targets)
            train_batches += 1.
        
            #print accurarcy on training sample:
            _, acc, mcc = fnn_validate(inputs, targets)
            train_acc += acc
            train_mcc += mcc
        
            #debugging output
            print 'train (',int(train_batches),'):    loss = ', train_err/train_batches,'\n',                                                '\t\tacc = ', train_acc/train_batches*100.,'\n',                                                 '\t\tmcc = ',train_mcc/train_batches
        
        # And a full pass over the validation data:
        val_err = 0.
        val_acc = 0.
        val_mcc = 0.
        val_batches = 0.
        for batch in hditer_validation.next_batch(batchsize):
            inputs, targets = batch            
            err, acc, mcc = fnn_validate(inputs, targets)
            val_err += err
            val_acc += acc
            val_mcc += mcc
            val_batches += 1.
        
        #save model if the validation error decreased
        if best_val_err>val_err:
            #write updates
            values1 = [p.get_value() for p in updates.keys()]
            with open(updatesfile_savebest,'wb+') as f:
                pickle.dump(values1,f)
                f.close()
        
            #write parameters
            values2 = ls.layers.get_all_param_values(network,trainable=True)
            with open(paramsfile_savebest,'wb+') as f:
                pickle.dump(values2,f)
                f.close()
            #update last error:
            best_val_err=val_err
        
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100.))
        print("  training mcc:\t\t{:.2f}".format(train_mcc / train_batches ))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100.))
        print("  validation mcc:\t\t{:.2f}".format(val_mcc / val_batches ))


# ## Store network


save_model=True
paramsfile_save="model_parameters.pick"
updatesfile_save="model_updates.pick"



if save_model:
    #write updates
    values1 = [p.get_value() for p in updates.keys()]
    with open(updatesfile_save,'wb+') as f:
        pickle.dump(values1,f)
        f.close()
        
    #write parameters
    values2 = ls.layers.get_all_param_values(network,trainable=True)
    with open(paramsfile_save,'wb+') as f:
        pickle.dump(values2,f)
        f.close()


# ## Compute performance metrics


from sklearn import metrics

#run on test data and compute ROC:
test_err = 0.
test_acc = 0.
test_batches = 0
batchsize_test=100

targets_pred = np.zeros((hditer_test.num_examples,))
targets_gt = np.zeros((hditer_test.num_examples,))

for batch in hditer_test.next_batch(batchsize_test):
    inputs, targets = batch
    err, acc, _ = fnn_validate(inputs,targets)
    test_err+=err
    test_acc+=acc
    test_batches+=1
    
    targets_pred[(test_batches-1)*batchsize_test:test_batches*batchsize_test] = fnn_det(inputs)[:,1]
    targets_gt[(test_batches-1)*batchsize_test:test_batches*batchsize_test] = targets[:]

#accuracies
print("  test loss:\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100.))



#ROC curve
#ROC
fpr, tpr, thresholds = metrics.roc_curve(targets_gt, targets_pred, pos_label=1)
plt.figure()
lw = 2
#full curve
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr,tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('ROC_1400_850.png',dpi=300)

#zoomed-in
#plt.plot(fpr, tpr, color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr,tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 0.01])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="center right")
plt.savefig('ROC_1400_850_zoom.png',dpi=300)
pass



def plot_example(x):
    plt.imshow(np.log10(x).T,extent=[-3.15, 3.15, -5, 5], interpolation='none',aspect='auto', origin='low')
    plt.colorbar()

#for batch in hditer_validation:
#    inputs,targets=batch
#    plot_example(inputs[0,0,:,:])
#    break

for batch in hditer_train:
    inputs,targets=batch
    plot_example(inputs[0,0,:,:])
    break;

