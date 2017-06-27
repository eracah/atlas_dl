
import matplotlib; matplotlib.use("agg")


#os stuff
import os
import sys
import h5py as h5

#timing
import time

#numpy
import numpy as np

#tensorflow
import tensorflow as tf
import tensorflow.contrib.keras as tfk

#housekeeping
import scripts.networks.binary_classifier_tf as bc


# # Network Parameters


args={'input_shape': [None, 64, 64, 1], 
                      'validation_batch_size': 128,
                      'test_batch_size': 128,
                      'weight_decay': 0, #0.0001, 
                      'num_fc_units': 512,
                      'num_layers': 3,
                      'batch_norm': True,
                      'time': True,
                      'conv_params': dict(num_filters=128, 
                                       filter_size=3, padding='SAME', 
                                       activation=tf.nn.relu, 
                                       initializer=tfk.initializers.he_normal())
                     }


# ## Build Network and Functions


print("Building model")
variables, network = bc.build_cnn_model(args)
pred_fn, loss_fn, accuracy_fn, auc_fn = bc.build_functions(variables, network)


# ## Setup Iterators


#path
print("Setting up iterators")
inputpath = '/global/cscratch1/sd/tkurth/atlas_dl/data_delphes_final_64x64'
logpath = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_logs'
#test files
trainfiles = [inputpath+'/'+x for x in os.listdir(inputpath) if x.startswith('hep_test') and x.endswith('.hdf5')]
trainset=bc.DataSet(trainfiles[0:20])
#validation files
validationfiles = [inputpath+'/'+x for x in os.listdir(inputpath) if x.startswith('hep_valid') and x.endswith('.hdf5')]
validationset=bc.DataSet(validationfiles[0:20])



#determining which model to load:
modelpath = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_models/hep_classifier_models'
metafilelist = [modelpath+'/'+x for x in os.listdir(modelpath) if x.endswith('.meta')]
metafilelist.sort()
metafile = metafilelist[-1]
checkpoint = metafile.replace(".meta","")
print metafile


# # Test Model


#initialize session
print("Start test")

#restore graph
model_saver = tf.train.import_meta_graph(metafile)

with tf.Session() as sess:
    
    #create graph
    graph = tf.get_default_graph()
    
    # Add an op to initialize the variables.
    init_global_op = tf.global_variables_initializer()
    
    #initialize variables
    sess.run([init_global_op])
    
    #restore weights belonging to graph
    model_saver.restore(sess,tf.train.latest_checkpoint(modelpath))
    
    #extract variables
    images_ = graph.get_tensor_by_name("Placeholder:0")
    keep_prob_ = graph.get_tensor_by_name("Placeholder_1:0")
    weights_ = graph.get_tensor_by_name("Placeholder_2:0")
    labels_ = graph.get_tensor_by_name("Placeholder_3:0")
    
    #extract predictor
    prediction_fn = tf.get_collection('pred_fn')
    
    #do a full pass over the validation set:
    all_labels=[]
    all_weights=[]
    all_pred=[]
            
    #iterate over batches
    while True:
        #get next batch
        images,labels,normweights,weights = validationset.next_batch(args['validation_batch_size'])
        #set weights to 1:
        normweights[:] = 1.
        
        #compute prediction
        pred=sess.run(prediction_fn,
                        feed_dict={images_: images, 
                        labels_: labels, 
                        weights_: normweights, 
                        keep_prob_: 1.0})
        
        print pred
        break
        
        #append to big numpy array:
        all_labels.append(labels)
        all_weights.append(weights)
        all_pred.append(pred)
        
         #check if full pass done
        if validationset._epochs_completed>0:
            validationset.reset()
            break





