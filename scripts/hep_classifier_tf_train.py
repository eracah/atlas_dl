
import matplotlib; matplotlib.use("agg")


#os stuff
import os
import h5py as h5

#numpy
import numpy as np

#sklearn
from sklearn import metrics

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
        self._images = np.transpose(self._images,(0,3,2,1))
        #select one channel only
        self._images = self._images[:,:,:,0:1]
        
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


# ## Model


def build_cnn_model(args):
    
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
        variables['conv'+str(layerid)+'_w']=tf.Variable(initializer([ksize,ksize,prev_num_filters,num_filters]),
                                                        name='conv'+str(layerid)+'_w')
        prev_num_filters=num_filters
        
        #conv unit
        network.append(tf.nn.conv2d(network[-1],
                                    filter=variables['conv'+str(layerid)+'_w'],
                                    strides=[1, 1, 1, 1], 
                                    padding=padding, 
                                    name='conv'+str(layerid)))
        
        outshape=network[-1].shape[1:]
        if args['batch_norm']:
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
        
        #add relu unit:
        network.append(activation(network[-1]))
        
        #add dropout
        network.append(tf.nn.dropout(network[-1],
                                     keep_prob=variables['keep_prob_'],
                                     name='drop'+str(layerid)))
        
        #add maxpool
        network.append(tf.nn.max_pool(network[-1],
                                      ksize=[1,2,2,1],
                                      strides=[1,2,2,1],
                                      padding=args['conv_params']['padding'],
                                      name='maxpool'+str(layerid)))
    
    #reshape
    network.append(tf.reshape(network[-1],shape=[-1, 8 * 8 * num_filters],name='flatten'))
    
    #now do the MLP
    #fc1
    variables['fc1_w']=tf.Variable(initializer([8 * 8 * num_filters,args['num_fc_units']]),name='fc1_w')
    variables['fc1_b']=tf.Variable(tf.zeros([args['num_fc_units']]),name='fc1_b')
    network.append(tf.matmul(network[-1], variables['fc1_w']) + variables['fc1_b'])
    
    #dropout
    network.append(tf.nn.dropout(network[-1],
                                     keep_prob=variables['keep_prob_'],
                                     name='drop'+str(layerid)))
    #fc2
    variables['fc2_w']=tf.Variable(initializer([args['num_fc_units'],2]),name='fc2_w')
    variables['fc2_b']=tf.Variable(tf.zeros([2]),name='fc2_b')
    network.append(tf.matmul(network[-1], variables['fc2_w']) + variables['fc2_b'])
    
    #softmax
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
    
    #compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(variables['labels_'],
                                                  prediction,
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


# # Network Parameters


args={'input_shape': [None, 64, 64, 1], 
                      'save_interval': 5,
                      'learning_rate': 1.e-6, 
                      'dropout_p': 0.5, 
                      'weight_decay': 0, #0.0001, 
                      'num_fc_units': 512,
                      'num_layers': 3,
                      'momentum': 0.9,
                      'num_epochs': 200,
                      'train_batch_size': 512, #480
                      'validation_batch_size': 320, #480
                      'batch_norm': True,
                      'conv_params': dict(num_filters=128, 
                                       filter_size=3, padding='SAME', 
                                       activation=tf.nn.relu, 
                                       initializer=tfk.initializers.he_normal())
                     }


# ## Build Network and Functions


print("Building model")
variables, network = build_cnn_model(args)
pred_fn, loss_fn, accuracy_fn, auc_fn = build_functions(variables, network)


# ## Setup Iterators


#path
print("Setting up iterators")
inputpath = '/global/cscratch1/sd/tkurth/atlas_dl/data_delphes_final_64x64'
logpath = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_logs'
modelpath = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_models'
#training files
trainfiles = [inputpath+'/'+x for x in os.listdir(inputpath) if x.startswith('hep_train') and x.endswith('.hdf5')]
trainset=DataSet(trainfiles[0:20])
#validation files
validationfiles = [inputpath+'/'+x for x in os.listdir(inputpath) if x.startswith('hep_valid') and x.endswith('.hdf5')]
validationset=DataSet(validationfiles[0:20])


# # Train Model


#initialize session
print("Start training")
with tf.Session() as sess:

    #train on training loss
    train_step = tf.train.AdamOptimizer(args['learning_rate']).minimize(loss_fn)

    #create summaries
    var_summary = []
    for item in variables:
        var_summary.append(tf.summary.histogram(item,variables[item]))
        #if item.startswith('conv'):
        #    #add additional image feature maps
        #    for i in range(variables_dict.shape[])
        #    tf.summary.image()
    summary_loss = tf.summary.scalar("loss",loss_fn)
    summary_accuracy = tf.summary.scalar("accuracy",accuracy_fn)
    train_summary = tf.summary.merge([summary_loss]+var_summary)
    validation_summary = tf.summary.merge([summary_loss])
    train_writer = tf.summary.FileWriter(logpath+'/hep_classifier_log', sess.graph)
    
    # Add an op to initialize the variables.
    init_global_op = tf.global_variables_initializer()
    init_local_op = tf.local_variables_initializer()
    
    #saver class:
    model_saver = tf.train.Saver()
    
    #initialize variables
    sess.run([init_global_op,init_local_op])
    
    #counter stuff
    epochs_completed=0
    trainset.reset()
    validationset.reset()
    train_loss=0.
    train_batches=0
    total_batches=0
    
    #do training
    while epochs_completed < args['num_epochs']:
        
        #increment total batch counter
        total_batches+=1
        
        #get next batch
        images,labels,normweights,_ = trainset.next_batch(args['train_batch_size'])  
    
        #update weights
        _, summary, tmp_loss = sess.run([train_step, train_summary, loss_fn],
                                           feed_dict={variables['images_']: images, 
                                              variables['labels_']: labels, 
                                              variables['weights_']: normweights, 
                                              variables['keep_prob_']: args['dropout_p']})
        
        #add to summary
        train_writer.add_summary(summary, total_batches)
        
        #increment train loss and batch number
        train_loss += tmp_loss
        train_batches += 1

        #check if epoch is done
        if trainset._epochs_completed>epochs_completed:
            epochs_completed=trainset._epochs_completed
            print("epoch %d, average training loss %g"%(epochs_completed, train_loss/float(train_batches)))
            train_loss=0.
            train_batches=0
            
            #compute validation loss:
            #reset variables
            validation_loss=0.
            validation_batches=0
            sess.run(init_local_op)
            
            all_labels=[]
            all_weights=[]
            all_pred=[]
            
            #iterate over batches
            while True:
                #get next batch
                images,labels,normweights,weights = validationset.next_batch(args['validation_batch_size'])
                #compute loss
                summary, tmp_loss=sess.run([validation_summary,loss_fn],
                                            feed_dict={variables['images_']: images, 
                                                        variables['labels_']: labels, 
                                                        variables['weights_']: normweights, 
                                                        variables['keep_prob_']: 1.0})
                
                #add loss
                validation_loss += tmp_loss
                validation_batches += 1
                
                #update accuracy
                sess.run(accuracy_fn[1],feed_dict={variables['images_']: images, 
                                                    variables['labels_']: labels, 
                                                    variables['weights_']: weights, 
                                                    variables['keep_prob_']: 1.0})
                
                #update auc
                sess.run(auc_fn[1],feed_dict={variables['images_']: images, 
                                              variables['labels_']: labels, 
                                              variables['weights_']: weights, 
                                              variables['keep_prob_']: 1.0})
                
                #debugging
                #pred = sess.run(pred_fn,
                #                feed_dict={variables['images_']: images, 
                #                            variables['labels_']: labels, 
                #                            variables['weights_']: weights, 
                #                            variables['keep_prob_']: 1.0})
                #all_labels.append(labels)
                #all_weights.append(weights)
                #all_pred.append(pred[:,1])
                
                #check if full pass done
                if validationset._epochs_completed>0:
                    validationset.reset()
                    break
                    
            
            #sklearn ROC
            #all_labels = np.concatenate(all_labels,axis=0).flatten()
            #all_pred = np.concatenate(all_pred,axis=0).flatten()
            #all_weights = np.concatenate(all_weights,axis=0).flatten()
            #fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_pred, pos_label=1, sample_weight=all_weights)
            #print("epoch %d, sklearn AUC %g"%(epochs_completed,metrics.auc(fpr,tpr,reorder=True)))
            
            print("epoch %d, average validation loss %g"%(epochs_completed, validation_loss/float(validation_batches)))
            validation_accuracy = sess.run(accuracy_fn[0])
            print("epoch %d, average validation accu %g"%(epochs_completed, validation_accuracy))
            validation_auc = sess.run(auc_fn[0])
            print("epoch %d, average validation auc %g"%(epochs_completed, validation_auc))
            
            # Save the variables to disk.
            if epochs_completed%args['save_interval']==0:
                model_save_path = model_saver.save(sess, modelpath+'/hep_classifier_tfmodel_epoch_'+str(epochs_completed)+'.ckpt')
                print 'Model saved in file: %s'%model_save_path





