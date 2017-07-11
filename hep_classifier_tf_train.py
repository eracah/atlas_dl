
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

#slurm helpers
sys.path.append("/global/homes/t/tkurth/python_custom_modules")
import slurm_tf_helper.setup_clusters as sc

#housekeeping
import scripts.networks.binary_classifier_tf as bc


# # Useful Functions


def train_loop(sess,train_step,args,trainset,validationset):
    
    #counter stuff
    trainset.reset()
    validationset.reset()
    
    #restore weights belonging to graph
    epochs_completed = 0
    if not args['restart']:
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
    while (epochs_completed < args['num_epochs']) and not sess.should_stop():
        
        #increment total batch counter
        total_batches+=1
        
        #get next batch
        images,labels,normweights,_ = trainset.next_batch(args['train_batch_size'])  
        #set weights to zero
        normweights[:] = 1.
        
        #update weights
        start_time = time.time()
        if args['create_summary']:
            _, summary, tmp_loss, pred = sess.run([train_step, train_summary, loss_fn, pred_fn],
                                                  feed_dict={variables['images_']: images, 
                                                  variables['labels_']: labels, 
                                                  variables['weights_']: normweights, 
                                                  variables['keep_prob_']: args['dropout_p']})
        else:
            _, tmp_loss, pred = sess.run([train_step, loss_fn, pred_fn],
                                        feed_dict={variables['images_']: images, 
                                        variables['labels_']: labels, 
                                        variables['weights_']: normweights, 
                                        variables['keep_prob_']: args['dropout_p']})
        end_time = time.time()
        train_time += end_time-start_time
        
        #add to summary
        if args['create_summary']:
            with tf.device(args['device']):
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
                if args['create_summary']:
                    summary, tmp_loss=sess.run([validation_summary,loss_fn],
                                                feed_dict={variables['images_']: images, 
                                                            variables['labels_']: labels, 
                                                            variables['weights_']: normweights, 
                                                            variables['keep_prob_']: 1.0})
                else:
                    tmp_loss=sess.run([loss_fn],
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
                model_save_path = model_saver.save(sess, args['modelpath']+'/hep_classifier_tfmodel_epoch_'+str(epochs_completed)+'.ckpt')
                print 'Model saved in file: %s'%model_save_path


# # Global Parameters


args={'input_shape': [None, 1, 64, 64], 
                      'arch' : 'knl',
                      'mode': "sync",
                      'num_tasks': 3,
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
                      'restart': False,
                      'conv_params': dict(num_filters=128, 
                                       filter_size=3, padding='SAME', 
                                       activation=tf.nn.relu, 
                                       initializer=tfk.initializers.he_normal())
                     }


# # On-Node Stuff


#common stuff
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

#arch-specific stuff
if args['arch']=='hsw':
    num_inter_threads = 2
    num_intra_threads = 16
elif args['arch']=='knl':
    num_inter_threads = 2
    num_intra_threads = 66
elif args['arch']=='k80':
    #use default settings
    p = tf.ConfigProto()
    num_inter_threads = int(getattr(p,'INTER_OP_PARALLELISM_THREADS_FIELD_NUMBER'))
    num_intra_threads = int(getattr(p,'INTRA_OP_PARALLELISM_THREADS_FIELD_NUMBER'))
else:
    raise ValueError('Please specify a valid architecture with arch (allowed values: hsw, knl.)')

#set the rest
os.environ['OMP_NUM_THREADS'] = str(num_intra_threads)
sess_config=tf.ConfigProto(inter_op_parallelism_threads=num_inter_threads,
                           intra_op_parallelism_threads=num_intra_threads,
                           allow_soft_placement=True, 
                           log_device_placement=True)

print("Using ",num_inter_threads,"-way task parallelism with ",num_intra_threads,"-way data parallelism.")


# # Multi-Node Stuff


#decide who will be worker and who will be parameters server
args['create_summary']=True
if args['num_tasks'] > 1:
    args['cluster'], args['server'], args['task_index'], args['num_tasks'], args['node_type'] = sc.setup_slurm_cluster(num_ps=1)
    if args['node_type'] == "ps":
        args['server'].join()
    elif args['node_type'] == "worker":
        args['is_chief']=(args['task_index'] == 0)
else:
    args['node_type']="worker"
    args['num_tasks']=1
    args['task_index']=1
    args['is_chief']=True


# ## Build Network and Functions


print("Building model")
if args['node_type'] == 'worker':
    args['device'] = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % args['task_index'],cluster=args['cluster'])
    with tf.device(args['device']):
        variables, network = bc.build_cnn_model(args)
        pred_fn, loss_fn, accuracy_fn, auc_fn = bc.build_functions(variables, network)
        tf.add_to_collection('pred_fn', pred_fn)
        tf.add_to_collection('loss_fn', loss_fn)
        print variables
        print network


# ## Setup Iterators

# ### My Files


if False and (args['node_type'] == 'worker'):
    print("Setting up iterators")
    #paths
    args['inputpath'] = '/global/cscratch1/sd/tkurth/atlas_dl/data_delphes_final_64x64'
    args['logpath'] = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_logs/hep_classifier_log'
    args['modelpath'] = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_models/hep_classifier_models'
    #training files
    trainfiles = [args['inputpath']+'/'+x for x in os.listdir(args['inputpath']) if x.startswith('hep_train') and x.endswith('.hdf5')]
    trainset=bc.DataSet(trainfiles,args['num_tasks'],args['task_index'])
    #validation files
    validationfiles = [args['inputpath']+'/'+x for x in os.listdir(args['inputpath']) if x.startswith('hep_valid') and x.endswith('.hdf5')]
    validationset = bc.DataSet(validationfiles,args['num_tasks'],args['task_index'])


# ### Evans Files


if True and (args['node_type'] == 'worker'):
    print("Setting up iterators")
    #paths
    args['inputpath'] = '/global/cscratch1/sd/wbhimji/delphes_combined_64imageNoPU'
    args['logpath'] = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_logs/hep_classifier_log'
    args['modelpath'] = '/project/projectdirs/mpccc/tkurth/MANTISSA-HEP/atlas_dl/temp/tensorflow_models/hep_classifier_models_distributed'
    #training files
    trainfiles = [args['inputpath']+'/'+x for x in os.listdir(args['inputpath']) if x.startswith('train_') and x.endswith('.h5')]
    trainset = bc.DataSetEvan(trainfiles,args['num_tasks'],args['task_index'],split_filelist=False,split_file=True)
    #validation files
    validationfiles = [args['inputpath']+'/'+x for x in os.listdir(args['inputpath']) if x.startswith('val_') and x.endswith('.h5')]
    validationset = bc.DataSetEvan(validationfiles,args['num_tasks'],args['task_index'],split_filelist=False,split_file=True)


# # Train Model


#determining which model to load:
metafilelist = [args['modelpath']+'/'+x for x in os.listdir(args['modelpath']) if x.endswith('.meta')]
if not metafilelist:
    #no model found, restart from scratch
    args['restart']=True



#initialize session
if (args['node_type'] == 'worker'):
    
    with tf.device(args['device']):
        
        #a hook that will stop training at
        hooks=[tf.train.StopAtStepHook(last_step=1000000)]
        
        #global step that either gets updated after any node processes a batch (async) or when all nodes process a batch for a given iteration (sync)
        global_step = tf.contrib.framework.get_or_create_global_step()     
        opt = tf.train.AdamOptimizer(args['learning_rate'])
        if args['mode'] == "sync":
            #if syncm we make a data structure that will aggregate the gradients form all tasks (one task per node in thsi case)
            opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=args['num_tasks'], total_num_replicas=args['num_tasks'])
        train_step = opt.minimize(loss_fn, global_step=global_step)
        
        if args["mode"] == "sync":
            hooks.append(opt.make_session_run_hook(is_chief=args['is_chief']))
            
        #creating summary
        if args['create_summary']:
            var_summary = []
            for item in variables:
                var_summary.append(tf.summary.histogram(item,variables[item]))
            summary_loss = tf.summary.scalar("loss",loss_fn)
            summary_accuracy = tf.summary.scalar("accuracy",accuracy_fn)
            train_summary = tf.summary.merge([summary_loss]+var_summary)
            validation_summary = tf.summary.merge([summary_loss])
            
        # Add an op to initialize the variables.
        init_global_op = tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
        
        #saver class:
        model_saver = tf.train.Saver()
        
    
    print("Start training")
    with tf.train.MonitoredTrainingSession(config=sess_config, 
                                           is_chief=args["is_chief"],
                                           master=args['server'].target,
                                           checkpoint_dir=args['modelpath'],
                                           hooks=hooks) as sess:
    
        #create summaries
        if args['create_summary']:
            with tf.device(args['device']):
                train_writer = tf.summary.FileWriter(args['logpath'], sess.graph)
        #        var_summary = []
        #        for item in variables:
        #            var_summary.append(tf.summary.histogram(item,variables[item]))
        #        summary_loss = tf.summary.scalar("loss",loss_fn)
        #        summary_accuracy = tf.summary.scalar("accuracy",accuracy_fn)
        #        train_summary = tf.summary.merge([summary_loss]+var_summary)
        #        validation_summary = tf.summary.merge([summary_loss])
        #        train_writer = tf.summary.FileWriter(args['logpath'], sess.graph)
    
        #initialize variables
        sess.run([init_global_op, init_local_op])
    
        #do the training loop
        train_loop(sess,train_step,args,trainset,validationset)





