
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


def load_data(bg_cfg_file = '../config/BgFileListAug16.txt',
                sig_cfg_file='../config/SignalFileListAug16.txt',
                num_files=10,  
                group_name='CollectionTree',
                branches=['CaloCalTopoClustersAuxDyn.calPhi', \
                          'CaloCalTopoClustersAuxDyn.calEta', \
                          'CaloCalTopoClustersAuxDyn.calE'],
                dataset_name='histo',
                type_='root'):

    #get list of files
    bg_files = [line.rstrip() for line in open(bg_cfg_file)]
    sig_files = [line.rstrip() for line in open(sig_cfg_file)]
    
    #so we don't have annoying stderr messages
    with suppress_stdout_stderr():
            
        #bgarray has n_events groups of 3 parallel numpy arrays 
        #(each numpy within a group is of equal length and each array corresponds to phi, eta and the corresponding energy)
        bgarray = rnp.root2array(bg_files[:num_files],                                 treename=group_name,                                 branches=branches,                                 start=0,                                 warn_missing_tree=True)

        sigarray = rnp.root2array(sig_files[:num_files],                                treename=group_name,                                branches=branches,                                start=0,                                 warn_missing_tree=True)
        
    #create dataframe with all entries
    #store in dataframe
    bgdf = pd.DataFrame.from_records(bgarray)
    bgdf['label']=0
    sigdf = pd.DataFrame.from_records(sigarray)
    sigdf['label']=1
    
    #concat
    return pd.concat([bgdf,sigdf])


#preprocessor
def preprocess_autoenc_data(df,num_resamplings,eta_range,phi_range,eta_bins,phi_bins):
    #empty array
    xvals = np.zeros((df.shape[0]*num_resamplings, 1, phi_bins, eta_bins ),dtype='float32')
    yvals = np.zeros((df.shape[0]*num_resamplings,),dtype='int32')
    
    for i in range(df.shape[0]):        
        phi, eta, E =  df.iloc[i]['CaloCalTopoClustersAuxDyn.calPhi'],                       df.iloc[i]['CaloCalTopoClustersAuxDyn.calEta'],                       df.iloc[i]['CaloCalTopoClustersAuxDyn.calE']
        
        start = i * num_resamplings
        stop = (i+1) * num_resamplings
        
        #x is histogrammed
        xtmp = np.histogram2d(phi,eta,
                            bins=(phi_bins, eta_bins), \
                            weights=E,
                            range=[phi_range,eta_range])[0]
        xvals[start:stop] = xtmp
        
        #y is simple:
        ytmp = np.zeros((num_resamplings),dtype='int32')
        ytmp.fill(df.iloc[i]['label'])
        yvals[start:stop] = ytmp[:]
        
    return xvals, yvals


# ## Data iterator


class hep_autoenc_data_iterator:
    
    #class constructor
    def __init__(self,
                 datadf,
                 num_resamplings,
                 max_frequency=None,
                 shuffle=True,
                 bin_size=0.025,
                 eta_range = [-5,5],
                 phi_range = [-3.14, 3.14]
                ):

        #set parameters
        self.num_resamplings = num_resamplings
        self.shuffle = shuffle
        self.bin_size = bin_size
        self.eta_range = eta_range
        self.phi_range = phi_range
        
        #compute bins
        self.eta_bins = int(np.floor((self.eta_range[1] - self.eta_range[0]) / self.bin_size))
        self.phi_bins = int(np.floor((self.phi_range[1] - self.phi_range[0]) / self.bin_size))
        
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
        tmpdf=self.df.groupby(['label']).apply(lambda x: x[['CaloCalTopoClustersAuxDyn.calPhi',                                                             'CaloCalTopoClustersAuxDyn.calEta',                                                             'CaloCalTopoClustersAuxDyn.calE']].iloc[:min_frequency,:]).copy()
        tmpdf.reset_index(inplace=True)
        del tmpdf['level_1']
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
    def next_batch(self,batchsize,num_units_latent):
        '''batch iterator'''
        
        #shuffle:
        if self.shuffle:
            self.df=self.df.reindex(np.random.permutation(self.df.index))
        
        #iterate
        for idx in range(0,self.num_examples-batchsize,batchsize):
            #yield next batch
            x,y=preprocess_autoenc_data(self.df.iloc[idx:idx+batchsize,:], self.num_resamplings,                              self.eta_range,
                             self.phi_range,
                             self.eta_bins,self.phi_bins)
            #rescale x:
            x/=self.max_abs
            
            #compute epsilon
            eps=np.random.normal(size=(batchsize*self.num_resamplings,num_units_latent))
            
            #return result
            yield x,y,eps


# ## Construct data iterator


#parameters
train_fraction=0.8
binsize=0.1
numfiles=2
num_resamplings=4

#load data
datadf=load_data(num_files=numfiles)

#create views for different labels
sigdf=datadf[ datadf.label==1 ]
bgdf=datadf[ datadf.label==0 ]

#split the sets
num_bg_train=int(np.floor(bgdf.shape[0]*train_fraction))
traindf_bg=bgdf.iloc[:num_bg_train]
validdf_bg=bgdf.iloc[num_bg_train:]
validdf_sig=sigdf

#create iterators
hditer_train_bg=hep_autoenc_data_iterator(traindf_bg,num_resamplings,max_frequency=4000,bin_size=binsize)
hditer_validation_bg=hep_autoenc_data_iterator(validdf_bg,num_resamplings,max_frequency=1000,bin_size=binsize)
hditer_validation_sig=hep_autoenc_data_iterator(validdf_sig,num_resamplings,max_frequency=1000,bin_size=binsize)

#the preprocessing for the validation iterator has to be taken from the training iterator
hditer_validation_bg.max_abs=hditer_train_bg.max_abs
hditer_validation_sig.max_abs=hditer_train_bg.max_abs



print hditer_train_bg.num_examples
print hditer_validation_bg.num_examples
print hditer_validation_sig.num_examples


# # Variational Autoencoder

# ## Some useful definitions


#KL divergence
def KLdiv(z_mu, z_log_sigma):
    return -0.5 * (1. + 2. * z_log_sigma - z_mu**2 - T.exp(2. * z_log_sigma)).sum(1)


class ReparamLayer(ls.layers.MergeLayer):
    """Layer for reparametrization trick: order of parameters: eps, mu, log_std.
    Computes mu + sigma * eps, so the result ~N(mu,sigma) if eps~N(0,1)
    
    """
    def __init__(self, incomings, **kwargs):
        super(ReparamLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        return inputs[1] + inputs[0] * T.exp(inputs[2])

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0])

    
#class TileLayer(ls.layers.Layer):
#    """Layer for tiling a given layer accoding to reps:
#    """
#    def __init__(self, incoming, reps, **kwargs):
#        super(TileLayer, self).__init__(incoming, **kwargs)
#        self.reps = reps
#        
#    def get_output_for(self, input, **kwargs):
#        return T.tile(input,self.reps)
#
#    def get_output_shape_for(self, input_shape):
#        assert len(self.reps) == len(input_shape)
#        return (self.reps*input_shape)


# ## Construct autoencoder network


#some parameters
keep_prob=0.5
num_filters=128
num_units_latent=10
initial_learning_rate=0.005

#input layer
l_inp_data = ls.layers.InputLayer((None,hditer_train_bg.xshape[0],                                   hditer_train_bg.xshape[1],                                   hditer_train_bg.xshape[2]))
l_inp_eps = ls.layers.InputLayer((None,num_units_latent))

#conv layers
#encoder 1:
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

#encoder 2:
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

#encoder 3:
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

#encoder 4:
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
l_flatten = ls.layers.FlattenLayer(incoming=l_pool4,
                                outdim=2)

l_project = ls.layers.DenseLayer(incoming=l_flatten, 
                             num_units=num_units_latent, 
                             W=ls.init.GlorotUniform(np.sqrt(2./(1+0.01**2))), 
                             b=ls.init.Constant(0.0),
                             nonlinearity=ls.nonlinearities.LeakyRectify()
                            )

#sampling layer
l_z_mean = ls.layers.DenseLayer(incoming=l_project, 
                             num_units=num_units_latent, 
                             W=ls.init.GlorotUniform(np.sqrt(2./(1+0.01**2))), 
                             b=ls.init.Constant(0.0),
                             nonlinearity=ls.nonlinearities.identity
                            )
l_z_log_sigma = ls.layers.DenseLayer(incoming=l_project, 
                             num_units=num_units_latent, 
                             W=ls.init.GlorotUniform(np.sqrt(2./(1+0.01**2))), 
                             b=ls.init.Constant(0.0),
                             nonlinearity=ls.nonlinearities.identity
                            )
l_z = ReparamLayer(incomings=[l_inp_eps, l_z_mean, l_z_log_sigma])

#unproject
l_unproject = ls.layers.InverseLayer(incoming=l_z, layer=l_project)

#deflatten
#l_unflatten = ls.layers.ReshapeLayer(incoming=l_unproject,
#                                    shape=([0],1,l_drop4.output_shape[2],l_drop4.output_shape[3]))
l_unflatten = ls.layers.InverseLayer(incoming=l_unproject, layer=l_flatten)

#decoder 4
l_unpool4 = ls.layers.InverseLayer(incoming=l_unflatten,
                                   layer=l_pool4)
l_deconv4 = ls.layers.TransposedConv2DLayer(incoming=l_unpool4,
                                            num_filters=num_filters,
                                            filter_size=3,
                                            stride=(1,1),
                                            W=ls.init.HeUniform(),
                                            b=ls.init.Constant(0.),
                                            nonlinearity=ls.nonlinearities.LeakyRectify()
                                           )

#decoder 3
l_unpool3 = ls.layers.InverseLayer(incoming=l_deconv4,
                                   layer=l_pool3)
l_deconv3 = ls.layers.TransposedConv2DLayer(incoming=l_unpool3,
                                            num_filters=num_filters,
                                            filter_size=3,
                                            stride=(1,1),
                                            W=ls.init.HeUniform(),
                                            b=ls.init.Constant(0.),
                                            nonlinearity=ls.nonlinearities.LeakyRectify()
                                           )

#decoder 2
l_unpool2 = ls.layers.InverseLayer(incoming=l_deconv3,
                                   layer=l_pool2)
l_deconv2 = ls.layers.TransposedConv2DLayer(incoming=l_unpool2,
                                            num_filters=num_filters,
                                            filter_size=3,
                                            stride=(1,1),
                                            W=ls.init.HeUniform(),
                                            b=ls.init.Constant(0.),
                                            nonlinearity=ls.nonlinearities.LeakyRectify()
                                           )

#decoder 1
l_unpool1 = ls.layers.InverseLayer(incoming=l_deconv2,
                                   layer=l_pool1)
l_out = ls.layers.TransposedConv2DLayer(incoming=l_unpool1,
                                            num_filters=1,
                                            filter_size=3,
                                            stride=(1,1),
                                            W=ls.init.HeUniform(),
                                            b=ls.init.Constant(0.),
                                            nonlinearity=ls.nonlinearities.identity
                                           )

#network
network = [l_inp_data, l_inp_eps,
           l_conv1, l_pool1, l_drop1,
           l_conv2, l_pool2, l_drop2,
           l_conv3, l_pool3, l_drop3,
           l_conv4, l_pool4, l_drop4,
           l_project, l_flatten,
           l_z_mean, l_z_log_sigma, l_z,
           l_unproject, l_unflatten,
           l_unpool4, l_deconv4,
           l_unpool3, l_deconv3,
           l_unpool2, l_deconv2,
           l_unpool1, l_out
          ]

#variables
inp = l_inp_data.input_var
eps = l_inp_eps.input_var

#output from the whole network
l_pred_data = ls.layers.get_output(l_out, {l_inp_data: inp, l_inp_eps: eps})
l_pred_data_det = ls.layers.get_output(l_out, {l_inp_data: inp, l_inp_eps: eps}, deterministic=True)

#output for the sampling layer
l_out_z_mean = ls.layers.get_output(l_z_mean,{l_inp_data: inp})
l_out_z_log_sigma = ls.layers.get_output(l_z_log_sigma,{l_inp_data: inp})

#loss functions:
klloss = KLdiv(l_out_z_mean, l_out_z_log_sigma).mean()
celoss = ls.objectives.squared_error(l_pred_data,inp).mean()
celoss_det = ls.objectives.squared_error(l_pred_data_det,inp).mean()
loss = celoss + klloss
loss_det = celoss_det

#parameters
params = ls.layers.get_all_params(network, trainable=True)

#updates
updates = ls.updates.adam(loss, params, learning_rate=initial_learning_rate)

#compile network function
fnn = theano.function([inp,eps], l_pred_data_det)
#training function to minimize
fnn_train = theano.function([inp,eps], loss, updates=updates)
#validation function with accuracy
fnn_validate = theano.function([inp,eps], loss_det)
#generator
#fnn_generate = theano.function([eps], ls.layers.get_output(l_out, {l_inp_eps: eps}, deterministic=True))


# ## Train autoencoder


num_epochs=10
batchsize=128

for epoch in range(num_epochs):
    train_err = 0.
    train_batches = 0.
    start_time = time.time()
    for batch in hditer_train_bg.next_batch(batchsize,num_units_latent):
        inputs, targets, epsilon = batch        
        train_err += fnn_train(inputs,epsilon)
        train_batches += 1.
        
        #debugging output
        print 'train: ', train_err/train_batches
        
    # And a full pass over the validation data for background and signal:
    val_bg_err = 0.
    val_bg_batches = 0.
    for batch in hditer_validation_bg.next_batch(batchsize,num_units_latent):
        inputs, targets, epsilon = batch
        val_bg_err += fnn_validate(inputs,epsilon)
        val_bg_batches += 1.
    
    val_sig_err = 0.
    val_sig_batches = 0.
    for batch in hditer_validation_sig.next_batch(batchsize,num_units_latent):
        inputs, targets, epsilon = batch
        val_sig_err += fnn_validate(inputs,epsilon)
        val_sig_batches += 1.

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss (bg):\t\t{:.6f}".format(val_bg_err / val_bg_batches))
    print("  validation loss (sig):\t\t{:.6f}".format(val_sig_err / val_sig_batches))


# ## RNG Debugger


mu = np.zeros((4,10))
mu[:]=1.
std = np.zeros((4,10))
std[:]=3.

z0 = np.zeros((4,10)) 
z1 = np.zeros((4,10))
z1[:,:] = 1.
eps=T.matrix('eps')
zmean0 = theano.shared(z0, name='zmean0')
zstd0 = theano.shared(z1, name='zstd0')
zmean = theano.shared(mu, name='zmean')
zstd = theano.shared(std, name='zstd')
zrng = zmean + zstd * eps #Trng.normal(size=zmean0.shape, avg=zmean0, std=zstd0)
zrng2 = zmean + zstd * Trng.normal(size=zmean0.shape)
f_theano = theano.function([eps], zrng)
f_theano2 = theano.function([], zrng2)

zmt=T.matrix('zmt')
zst=T.matrix('zst')
f_KL = theano.function([zmt,zst],KLdiv(zmt,zst))



arr=[]
arr2=[]
for n in range(10000):
    eps=np.random.normal(loc=0., scale=1., size=(4,10))
    arr.append(f_theano(eps)[0,0])
    arr2.append(f_theano2()[0,0])
n, bins, patches = plt.hist(arr, 100, normed=1, facecolor='green', alpha=0.75)
y = mlab.normpdf( bins, 1, 3)
l = plt.plot(bins, y, 'r--', linewidth=1)
pass





