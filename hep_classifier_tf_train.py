
import matplotlib; matplotlib.use("agg")


#os stuff
import os
import sys
import h5py as h5
import re

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


args={'input_shape': [None, 1, 64, 64], 
                      'arch' : 'hsw',
                      'display_interval': 10,
                      'save_interval': 1,
                      'learning_rate': 1.e-5, 
                      'dropout_p': 0.5, 
                      'weight_decay': 0, #0.0001, 
                      'num_fc_units': 512,
                      'num_layers': 3,
                      'momentum': 0.9,
                      'num_epochs': 200,
                      'train_batch_size': 512, #480
                      'validation_batch_size': 320, #480
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
tf.add_to_collection('pred_fn', pred_fn)
tf.add_to_collection('loss_fn', loss_fn)



print variables
print network


# ## Setup Iterators

# ### My Files


if False:
    print("Setting up iterators")
    #paths
    inputpath = '/global/cscratch1/sd/tkurth/atlas_dl/data_delphes_final_64x64'
    logpath = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_logs/hep_classifier_log'
    modelpath = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_models/hep_classifier_models'
    #training files
    trainfiles = [inputpath+'/'+x for x in os.listdir(inputpath) if x.startswith('hep_train') and x.endswith('.hdf5')]
    trainset=bc.DataSet(trainfiles[0:20])
    #validation files
    validationfiles = [inputpath+'/'+x for x in os.listdir(inputpath) if x.startswith('hep_valid') and x.endswith('.hdf5')]
    validationset = bc.DataSet(validationfiles[0:20])


# ### Evans Files


if True:
    print("Setting up iterators")
    #paths
    inputpath = '/global/cscratch1/sd/wbhimji/delphes_combined_64imageNoPU'
    logpath = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_logs/hep_classifier_log'
    modelpath = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_models/hep_classifier_models'
    #training files
    trainfiles = [inputpath+'/'+x for x in os.listdir(inputpath) if x.startswith('train_') and x.endswith('.h5')]
    trainset=bc.DataSetEvan(trainfiles)
    #validation files
    validationfiles = [inputpath+'/'+x for x in os.listdir(inputpath) if x.startswith('val_') and x.endswith('.h5')]
    validationset = bc.DataSetEvan(validationfiles)


# # Train Model


arch=args['arch']

#common stuff
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

#arch-specific stuff
if arch=='hsw':
    num_inter_threads = 2
    num_intra_threads = 16
elif arch=='knl':
    num_inter_threads = 2
    num_intra_threads = 66
elif arch=='k80':
    num_inter_threads = -1
    num_intra_threads = -1
else:
    raise ValueError('Please specify a valid architecture with arch (allowed values: hsw, knl.)')

#set the rest
os.environ['OMP_NUM_THREADS'] = str(num_intra_threads)
sess_config=tf.ConfigProto(inter_op_parallelism_threads=num_inter_threads,
                           intra_op_parallelism_threads=num_intra_threads,
                           allow_soft_placement=True, 
                           log_device_placement=True)

print("Using ",num_inter_threads,"-way task parallelism with ",num_intra_threads,"-way data parallelism.")



restart=False
#determining which model to load:
metafilelist = [modelpath+'/'+x for x in os.listdir(modelpath) if x.endswith('.meta')]
if not metafilelist:
    restart=True
#metafilelist.sort()
#metafile = metafilelist[-1]
#checkpoint = metafile.replace(".meta","")
#print metafile
#restart from scratch or restore?



#initialize session
print("Start training")
with tf.Session(config=sess_config) as sess:

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
    train_writer = tf.summary.FileWriter(logpath, sess.graph)
    
    # Add an op to initialize the variables.
    init_global_op = tf.global_variables_initializer()
    init_local_op = tf.local_variables_initializer()
    
    #saver class:
    model_saver = tf.train.Saver()        
    
    #initialize variables
    sess.run([init_global_op, init_local_op])
    
    #counter stuff
    trainset.reset()
    validationset.reset()
    
    #restore weights belonging to graph
    epochs_completed = 0
    if not restart:
        last_model = tf.train.latest_checkpoint(modelpath)
        print("Restoring model %s.",last_model)
        model_saver.restore(sess,last_model)
        epochs_completed = int(re.match(r'^.*?\_epoch\_(.*?)\.ckpt.*?$',last_model).groups()[0])
        trainset._epochs_completed = epochs_completed
    
    #losses
    train_loss=0.
    train_batches=0
    total_batches=0
    train_time=0
    
    #do training
    while epochs_completed < args['num_epochs']:
        
        #increment total batch counter
        total_batches+=1
        
        #get next batch
        images,labels,normweights,_ = trainset.next_batch(args['train_batch_size'])  
        #set weights to zero
        normweights[:] = 1.
        
        #update weights
        start_time = time.time()
        _, summary, tmp_loss, pred = sess.run([train_step, train_summary, loss_fn, pred_fn],
                                           feed_dict={variables['images_']: images, 
                                              variables['labels_']: labels, 
                                              variables['weights_']: normweights, 
                                              variables['keep_prob_']: args['dropout_p']})
        end_time = time.time()
        train_time += end_time-start_time
        
        #add to summary
        train_writer.add_summary(summary, total_batches)
        
        #increment train loss and batch number
        train_loss += tmp_loss
        train_batches += 1
        
        #determine if we give a short update:
        if train_batches%args['display_interval']==0:
            print("REPORT epoch %d.%d, average training loss %g (%.3f sec/batch)"%(epochs_completed, train_batches,
                                                                                train_loss/float(train_batches),
                                                                                train_time/float(train_batches)))
        
        #check if epoch is done
        if trainset._epochs_completed>epochs_completed:
            epochs_completed=trainset._epochs_completed
            print("COMPLETED epoch %d, average training loss %g (%.3f sec/batch)"%(epochs_completed, 
                                                                                 train_loss/float(train_batches),
                                                                                 train_time/float(train_batches)))
            train_loss=0.
            train_batches=0
            train_time=0
            
            #compute validation loss:
            #reset variables
            validation_loss=0.
            validation_batches=0
            sess.run(init_local_op)
            
            #all_labels=[]
            #all_weights=[]
            #all_pred=[]
            
            #iterate over batches
            while True:
                #get next batch
                images,labels,normweights,weights = validationset.next_batch(args['validation_batch_size'])
                #set weights to 1:
                normweights[:] = 1.
                weights[:] = 1.
                
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
            
            print("COMPLETED epoch %d, average validation loss %g"%(epochs_completed, validation_loss/float(validation_batches)))
            validation_accuracy = sess.run(accuracy_fn[0])
            print("COMPLETED epoch %d, average validation accu %g"%(epochs_completed, validation_accuracy))
            validation_auc = sess.run(auc_fn[0])
            print("COMPLETED epoch %d, average validation auc %g"%(epochs_completed, validation_auc))
            
            # Save the variables to disk.
            if epochs_completed%args['save_interval']==0:
                model_save_path = model_saver.save(sess, modelpath+'/hep_classifier_tfmodel_epoch_'+str(epochs_completed)+'.ckpt')
                print 'Model saved in file: %s'%model_save_path





