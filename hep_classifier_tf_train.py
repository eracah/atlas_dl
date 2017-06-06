
import matplotlib; matplotlib.use("agg")


#os stuff
import os
import sys
import h5py as h5

#numpy
import numpy as np

#tensorflow
import tensorflow as tf
import tensorflow.contrib.keras as tfk

import scripts.networks.binary_classifier_tf as bc


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
variables, network = bc.build_cnn_model(args)
pred_fn, loss_fn, accuracy_fn, auc_fn = bc.build_functions(variables, network)


# ## Setup Iterators


#path
print("Setting up iterators")
inputpath = '/global/cscratch1/sd/tkurth/atlas_dl/data_delphes_final_64x64'
logpath = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_logs'
modelpath = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_models'
#training files
trainfiles = [inputpath+'/'+x for x in os.listdir(inputpath) if x.startswith('hep_train') and x.endswith('.hdf5')]
trainset=bc.DataSet(trainfiles[0:20])
#validation files
validationfiles = [inputpath+'/'+x for x in os.listdir(inputpath) if x.startswith('hep_valid') and x.endswith('.hdf5')]
validationset=bc.DataSet(validationfiles[0:20])


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





