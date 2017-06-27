
import matplotlib; matplotlib.use("agg")


#os stuff
import os
import h5py as h5

#numpy
import numpy as np

#tensorflow
import tensorflow as tf
import tensorflow.contrib.keras as tfk


# # General Functions



class DataSet(object):
    
    def reset(self):
        self._epochs_completed = 0
        self._file_index = 0
        self._data_index = 0
    
    
    def load_next_file(self):
        with h5.File(self._filelist[self._file_index],'r') as f:
            self._images = f['data'].value
            self._labels = f['label'].value
            self._normweights = f['normweight'].value
            self._weights = f['weight'].value
            f.close()
        assert self._images.shape[0] == self._labels.shape[0], ('images.shape: %s labels.shape: %s' % (self._images.shape, self_.labels.shape))
        assert self._labels.shape[0] == self._normweights.shape[0], ('labels.shape: %s normweights.shape: %s' % (self._labels.shape, self._normweights.shape))
        
        #set number of samples
        self._num_examples = self._labels.shape[0]
        
        #create permutation
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        
        #shuffle
        self._images = self._images[perm]
        self._labels = self._labels[perm]
        self._normweights = self._normweights[perm]
        self._weights = self._weights[perm]
        
        #transpose images
        #self._images = np.transpose(self._images,(0,3,2,1))
        #select one channel only
        self._images = self._images[:,0:1,:,:]
        
        #reshape labels and weights
        self._labels = np.reshape(self._labels,(self._labels.shape[0],1))
        self._normweights = np.reshape(self._normweights,(self._normweights.shape[0],1))
        self._weights = np.reshape(self._weights,(self._weights.shape[0],1))
        
    
    def __init__(self, filelist):
        """Construct DataSet"""
        self._num_files = len(filelist)
        
        assert self._num_files > 0, ('filelist is empty')
        
        self._filelist = filelist
        self.reset()
        self.load_next_file()

    @property
    def num_files(self):
        return self._num_files
    
    @property
    def num_samples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._data_index
        self._data_index += batch_size
        if self._data_index > self._num_examples:
            
            #first, reset data_index and increase file index:
            start=0
            self._data_index=batch_size
            self._file_index+=1
            
            #check if we are at the end of the file list
            if self._file_index >= self._num_files:
                #epoch is finished
                self._epochs_completed += 1
                #reset file index and shuffle list
                self._file_index=0
                np.random.shuffle(self._filelist)
            
            #load the next file
            self.load_next_file()
            assert batch_size <= self._num_examples
        
        end = self._data_index
        return self._images[start:end], self._labels[start:end], self._normweights[start:end], self._weights[start:end]



class DataSetEvan(object):
    
    def reset(self):
        self._epochs_completed = 0
        self._file_index = 0
        self._data_index = 0
    
    
    def load_next_file(self):
        #only load a new file if there are more than one file in the list:
        if self._num_files > 1 or not self._initialized:
            with h5.File(self._filelist[self._file_index],'r') as f:
                self._images = f['all_events/hist'].value
                self._labels = f['all_events/y'].value
                self._normweights = f['all_events/normalized_weight'].value
                self._weights = f['all_events/weight'].value
                f.close()
                
            #sanity checks
            assert self._images.shape[0] == self._labels.shape[0], ('images.shape: %s labels.shape: %s' % (self._images.shape, self_.labels.shape))
            assert self._labels.shape[0] == self._normweights.shape[0], ('labels.shape: %s normweights.shape: %s' % (self._labels.shape, self._normweights.shape))
            self._initialized = True
        
            #set number of samples
            self._num_examples = self._labels.shape[0]
            
            #transpose images
            shape = self._images.shape
            self._images = np.reshape(self._images,(shape[0],1,shape[1],shape[2]))
        
            #reshape labels and weights
            self._labels = np.reshape(self._labels,(self._labels.shape[0],1))
            self._normweights = np.reshape(self._normweights,(self._normweights.shape[0],1))
            self._weights = np.reshape(self._weights,(self._weights.shape[0],1))
            
        #create permutation
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        #shuffle
        self._images = self._images[perm]
        self._labels = self._labels[perm]
        self._normweights = self._normweights[perm]
        self._weights = self._weights[perm]
        
    
    def __init__(self, filelist):
        """Construct DataSet"""
        self._num_files = len(filelist)
        
        assert self._num_files > 0, ('filelist is empty')
        
        self._filelist = filelist
        self._initialized = False
        self.reset()
        self.load_next_file()

    @property
    def num_files(self):
        return self._num_files
    
    @property
    def num_samples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._data_index
        self._data_index += batch_size
        if self._data_index > self._num_examples:
            
            #first, reset data_index and increase file index:
            start=0
            self._data_index=batch_size
            self._file_index+=1
            
            #check if we are at the end of the file list
            if self._file_index >= self._num_files:
                #epoch is finished
                self._epochs_completed += 1
                #reset file index and shuffle list
                self._file_index=0
                np.random.shuffle(self._filelist)
            
            #load the next file
            self.load_next_file()
            assert batch_size <= self._num_examples
        
        end = self._data_index
        return self._images[start:end], self._labels[start:end], self._normweights[start:end], self._weights[start:end]


# ## Model


def build_cnn_model(args):
    
    #find out which device to use:
    device='/cpu:0'
    if args['arch']=='k80':
        device='/gpu:0'
    
    #define empty variables dict
    variables={}
    
    #create placeholders
    variables['images_'] = tf.placeholder(tf.float32, shape=args['input_shape'])
    variables['keep_prob_'] = tf.placeholder(tf.float32)
    
    #empty network:
    network = []
    
    #input layer
    network.append(tf.reshape(variables['images_'], [-1]+args['input_shape'][1:], name='input'))
    
    #get all the conv-args stuff:
    activation=args['conv_params']['activation']
    initializer=args['conv_params']['initializer']
    ksize=args['conv_params']['filter_size']
    num_filters=args['conv_params']['num_filters']
    padding=args['conv_params']['padding']
        
    #conv layers:
    prev_num_filters=1
    for layerid in range(1,args['num_layers']+1):
        
        #create weight-variable
        with tf.device(device):
            variables['conv'+str(layerid)+'_w']=tf.Variable(initializer([ksize,ksize,prev_num_filters,num_filters]),
                                                            name='conv'+str(layerid)+'_w')
            prev_num_filters=num_filters
        
            #conv unit
            network.append(tf.nn.conv2d(network[-1],
                                        filter=variables['conv'+str(layerid)+'_w'],
                                        strides=[1, 1, 1, 1], 
                                        padding=padding,
                                        data_format="NCHW",
                                        name='conv'+str(layerid)))
        
        #batchnorm if desired
        outshape=network[-1].shape[1:]
        if args['batch_norm']:
            with tf.device(device):
                #mu
                variables['bn'+str(layerid)+'_m']=tf.Variable(tf.zeros(outshape),
                                                             name='bn'+str(layerid)+'_m')
                #sigma
                variables['bn'+str(layerid)+'_s']=tf.Variable(tf.ones(outshape),
                                                             name='bn'+str(layerid)+'_s')
                #gamma
                variables['bn'+str(layerid)+'_g']=tf.Variable(tf.ones(outshape),
                                                             name='bn'+str(layerid)+'_g')
                #beta
                variables['bn'+str(layerid)+'_b']=tf.Variable(tf.zeros(outshape),
                                                             name='bn'+str(layerid)+'_b')
                #add batch norm layer
                network.append(tf.nn.batch_normalization(network[-1],
                               mean=variables['bn'+str(layerid)+'_m'],
                               variance=variables['bn'+str(layerid)+'_s'],
                               offset=variables['bn'+str(layerid)+'_b'],
                               scale=variables['bn'+str(layerid)+'_g'],
                               variance_epsilon=1.e-4,
                               name='bn'+str(layerid)))
        
        #add relu unit
        with tf.device(device):
            network.append(activation(network[-1]))
        
        #add maxpool
        with tf.device(device):
            network.append(tf.nn.max_pool(network[-1],
                                          ksize=[1,1,2,2],
                                          strides=[1,1,2,2],
                                          padding=args['conv_params']['padding'],
                                          data_format="NCHW",
                                          name='maxpool'+str(layerid)))
        
        #add dropout
        with tf.device(device):
            network.append(tf.nn.dropout(network[-1],
                                         keep_prob=variables['keep_prob_'],
                                         name='drop'+str(layerid)))
    
    #reshape
    with tf.device(device):
        network.append(tf.reshape(network[-1],shape=[-1, 8 * 8 * num_filters],name='flatten'))
    
    #now do the MLP
    #fc1
    with tf.device(device):
        variables['fc1_w']=tf.Variable(initializer([8 * 8 * num_filters,args['num_fc_units']]),name='fc1_w')
        variables['fc1_b']=tf.Variable(tf.zeros([args['num_fc_units']]),name='fc1_b')
        network.append(tf.matmul(network[-1], variables['fc1_w']) + variables['fc1_b'])
    
    #add relu unit
    with tf.device(device):
        network.append(activation(network[-1]))
    
    #add dropout
    with tf.device(device):
        network.append(tf.nn.dropout(network[-1],
                                     keep_prob=variables['keep_prob_'],
                                     name='drop'+str(layerid)))
    #fc2
    with tf.device(device):
        variables['fc2_w']=tf.Variable(initializer([args['num_fc_units'],2]),name='fc2_w')
        variables['fc2_b']=tf.Variable(tf.zeros([2]),name='fc2_b')
        network.append(tf.matmul(network[-1], variables['fc2_w']) + variables['fc2_b'])
    
    #add softmax
    with tf.device(device):
        network.append(tf.nn.softmax(network[-1]))
    
    #return the network and variables
    return variables,network


#build the functions
def build_functions(variables, network):
    #add additional variables
    variables['labels_']=tf.placeholder(tf.int32,shape=[None,1])
    variables['weights_']=tf.placeholder(tf.float32,shape=[None,1])
    
    #loss function
    prediction = network[-1]
    tf.add_to_collection('prediction_op', prediction)
    
    #compute loss, important: use unscaled version!
    loss = tf.losses.sparse_softmax_cross_entropy(variables['labels_'],
                                                  network[-2],
                                                  weights=variables['weights_'])
    
    #compute accuracy
    accuracy = tf.metrics.accuracy(variables['labels_'],
                                   tf.round(prediction[:,1]),
                                   weights=variables['weights_'],
                                   name='accuracy')
    
    #compute AUC
    auc = tf.metrics.auc(variables['labels_'],
                         prediction[:,1],
                         weights=variables['weights_'],
                         num_thresholds=5000,
                         curve='ROC',
                         name='AUC')
    
    #get loss
    return prediction, loss, accuracy, auc





