from __future__ import print_function
import sys
sys.path.insert(0,'/project/projectdirs/mpccc/tkurth/NESAP/intelcaffe_internal/install_cori-hsw-nompi/python')
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import numpy as np
import os
import shutil
import stat
import subprocess
import re
import datetime as dt
import pandas as pd


#parse caffe output
#iteration results
def parse_iteration_results(outputbuffer):
    all_lines = []
    lines=outputbuffer.split("\n")

    #compile rules
    trainrule=re.compile(r'(.*)solver\.cpp(.*)Iteration(.*)loss(.*)')
    testrule=re.compile(r'^I.*?(\d{2}:\d{2}:\d{2}\.\d{6}).*?Test net output.*?loss\s=\s(.*?)\s\(')

    #iterate over lines
    lastiter=0
    for line in lines:

        #load train output
        if trainrule.match(line):
            try:
                (
                    _,time_field,pid,source_line,check_iteration,
                    niter,check_loss,check_equals,loss
                ) =line.split()
                assert(check_iteration=="Iteration")
                assert(check_loss=="loss")
                assert(check_equals=='=')
                loss=float(loss)
                niter=int(niter[:-1])
                difft = dt.datetime.strptime(time_field, "%H:%M:%S.%f")

                #append to dictionary
                new_line = {"time": difft, "iteration": niter, "loss": loss, "phase": "train"}
                lastiter=niter

                all_lines.append(new_line)
            except:
                continue

        #load test output
        testmatch=testrule.match(line)
        if testmatch:
            try:
                time_field=testmatch.groups()[0]
                loss=float(testmatch.groups()[1])
                niter=lastiter
                difft = dt.datetime.strptime(time_field, "%H:%M:%S.%f")
                new_line={"time": difft, "iteration": niter, "loss": loss, "phase": "validation"}
                all_lines.append(new_line)
            except:
                continue
    return all_lines


#preparing to run
#initialize folder structure
def create_folder_structure(expath):
    make_if_not_exist(expath)
    return


#create solver file:
def create_solver_file(params_spearmint,params_optimize,params_fixed):
    #set up solver parameters
    solver_param = {
        # Train parameters
        # different LR for ADAM. current optimum: 0.0001
        'base_lr': 10.**(params_optimize['base_lr'][0]),
        'weight_decay': 10.**(params_optimize['weight_decay'][0]),
        'lr_policy': "fixed",
        'iter_size': 1,
        'max_iter': params_fixed['max_iter'],
        'snapshot': params_fixed['iters_per_epoch'],
        'display': params_fixed['iters_per_epoch'],
        'average_loss': params_fixed['iters_per_epoch'],
        'type': "Adam",
        'solver_mode': P.Solver.CPU,
        'debug_info': False,
        'snapshot_after_train': True,
        'snapshot_format': P.Solver.HDF5,
        # Test parameters
        'test_iter': [params_fixed['num_iters_validation']],
        'test_interval': params_fixed['iters_per_epoch'],
        'eval_type': "classification",
        'test_initialization': True
    }

    # solver parameters
    solver = caffe_pb2.SolverParameter(
            train_net=params_spearmint['experiment_path']+'/'+params_spearmint['filename_train_net'],
            test_net=[params_spearmint['experiment_path']+'/'+params_spearmint['filename_validation_net']],
            snapshot_prefix=params_spearmint['experiment_path']+'/'+params_spearmint['filename_snapshot_prefix'],
            **solver_param)

    #save solver file
    with open(params_spearmint['experiment_path']+'/'+params_spearmint['filename_solver'], 'w') as f:
        print(solver, file=f)
        f.close()

    #return the solver
    return solver


#prepare network file
def create_network_file(params_spearmint,params_optimize,params_fixed):

    networks={}
    for ph in ["train", "validation"]:
        #phaseval
        phaseval=None
        if ph=="train":
            phaseval=caffe_pb2.TRAIN
        else:
            phaseval=caffe_pb2.TEST

        # dummy
        net = caffe.NetSpec()

        #data layer
        net.data, net.label = L.HDF5Data(name="data",
                                            include=dict(phase=phaseval),
                                            ntop=2,
                                            hdf5_data_param=dict(
                                                source=params_spearmint['experiment_path']+'/'+params_spearmint["filename_"+ph+"_data"],
                                                shuffle=1,
                                                batch_size=params_fixed["batch_size_"+ph]
                                                )
                                            )

        #conv-relu-pool-combinations
        from_layer=net.data
        weightinitpars=dict(weight_filler=dict(type="msra"),bias_filler=dict(type="constant",value=0))
        for n in range(1,params_fixed["num_conv"]+1):
            net["conv"+str(n)] = L.Convolution(from_layer, num_output=params_fixed["num_filters"],
                                                kernel_size=3, pad=1,
                                                stride=1, **weightinitpars)
            net["relu"+str(n)]=L.ReLU(net["conv"+str(n)], negative_slope=10.**(params_optimize["leakiness"][0]), in_place=True)
            if (n==params_fixed["num_conv"]):
                net["pool"+str(n)]=L.Pooling(net["relu"+str(n)], pool=P.Pooling.AVE, global_pooling=True)
            else:
                net["pool"+str(n)]=L.Pooling(net["relu"+str(n)], pool=P.Pooling.MAX, kernel_size=2, stride=2)
            from_layer=net["pool"+str(n)]

        #fully connected
        net["fc1"] = L.InnerProduct(from_layer, num_output=2, **weightinitpars)

        #loss
        net["loss"] = L.SoftmaxWithLoss(net["fc1"], net.label, include=dict(phase=phaseval))

        #save network
        with open(params_spearmint['experiment_path']+'/'+params_spearmint['filename_'+ph+'_net'], 'w') as f:
            print('engine: "MKL2017"', file=f)
            print('name: "{}_{}"'.format(params_fixed['network_name'],ph), file=f)
            print(net.to_proto(), file=f)
            f.close()

        #save network in dict
        networks[ph]=net

    #return the networks
    return networks


#set up data input files
def create_input_files(params_spearmint):
    #training
    with open(params_spearmint['experiment_path']+'/'+params_spearmint["filename_train_data"],"w") as f:
        print(params_spearmint["data_path"]+"/hep_training_chunk"+str(params_spearmint["fileindex"])+".hdf5",file=f)
        f.close()

    #validation
    with open(params_spearmint['experiment_path']+'/'+params_spearmint["filename_validation_data"],"w") as f:
        print(params_spearmint["data_path"]+"/hep_validation_chunk"+str(params_spearmint["fileindex"])+".hdf5",file=f)
        f.close()

        return


#set up runscript
def create_runscript(params_spearmint):
    with open(params_spearmint['experiment_path']+'/'+params_spearmint["filename_runscript"],"w") as f:
        #header
        print("#!/bin/bash\n\n",file=f)

        #OpenMP stuff
        print("export OMP_NUM_THREADS=66",file=f)
        print("export OMP_PLACES=threads",file=f)
        print("export OMP_PROC_BIND=spread\n\n",file=f)

        #executable
        print(params_spearmint["caffe_path"]+'/bin/caffe train --solver='+params_spearmint['experiment_path']+'/'+params_spearmint["filename_solver"]+" #> "+params_spearmint["experiment_path"]+"/train_val.out 2>&1",file=f)

        #put a wait at the end
        print("wait",file=f)

        #close file
        f.close()

        #make the script executable
        st = os.stat(params_spearmint['experiment_path']+'/'+params_spearmint["filename_runscript"])
        os.chmod(params_spearmint['experiment_path']+'/'+params_spearmint["filename_runscript"], st.st_mode | stat.S_IEXEC)

        return


#run training: use pycaffe
def run_training(params_spearmint,params_fixed):

    #set OpenMP
    os.environ["OMP_NUM_THREADS"] = str(params_fixed["num_threads"])
    os.environ["OMP_PLACES"] = "threads"
    os.environ["OMP_PROC_BIND"] = "spread"

    #get solver
    solver = caffe.get_solver(params_spearmint['experiment_path']+'/'+params_spearmint['filename_solver'])

    #compute number of epochs
    max_epochs=int(params_fixed["max_iter"]/params_fixed["iters_per_epoch"])
    patience_epochs=int(params_fixed["patience_iter"]/params_fixed["iters_per_epoch"])
    patience_increase_epochs=int(params_fixed["patience_increase_iter"]/params_fixed["iters_per_epoch"])

    #early stopping parameters
    best_validation_loss = np.inf
    patience=params_fixed["patience_iter"]

    #initialize loss arrays
    train_loss = []
    validation_loss = []
    #main solver loop
    epoch=0
    training_finished=False
    while (epoch < max_epochs) and not (training_finished):
        epoch+=1
        #train network
        solver.step(params_fixed["iters_per_epoch"])
        train_loss.append(solver.net.blobs['loss'].data.tolist())
        #validate network
        solver.test_nets[0].forward()
        validation_loss.append(solver.test_nets[0].blobs['loss'].data.tolist())
        #early stopping condition
        if validation_loss[-1] < best_validation_loss:
            #if improvement was good enough, improve patience
            if validation_loss[-1] < (best_validation_loss*params_fixed["improvement_threshold"]):
                patience_epochs = np.max([patience_epochs, epoch * patience_increase_epochs])
            best_validation_loss=validation_loss[-1]

        #check if we can top
        if patience_epochs<epoch:
            training_finished=True

    #return the best validation loss
    return best_validation_loss


#run distributed training: use the generated script
def run_distributed_training(params_spearmint,params_fixed):
    #run training
    p = subprocess.Popen(["/bin/bash",params_spearmint['experiment_path']+'/'+params_spearmint["filename_runscript"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate()
    p.wait()

    #parse output:
    resultdf=pd.DataFrame(parse_iteration_results(err))
    resultdf.sort_values(["phase","iteration"],inplace=True)
    validation_loss=resultdf["loss"].ix[ resultdf.phase == "validation" ].values

    #do "early"-stopping after the fact: determine the best loss depending on the learning curve improvement:
    #compute number of epochs
    max_epochs=int(params_fixed["max_iter"]/params_fixed["iters_per_epoch"])
    patience_epochs=int(params_fixed["patience_iter"]/params_fixed["iters_per_epoch"])
    patience_increase_epochs=int(params_fixed["patience_increase_iter"]/params_fixed["iters_per_epoch"])
    #early stopping parameters
    best_validation_loss = np.inf
    patience=params_fixed["patience_iter"]
    #main loop
    epoch=0
    training_finished=False
    while (epoch < max_epochs) and not (training_finished):
        epoch+=1
        if validation_loss[epoch-1] < best_validation_loss:
            if validation_loss[epoch-1] < (best_validation_loss*0.995):
                patience_epochs = np.max([patience_epochs, epoch * patience_increase_epochs])
            best_validation_loss=validation_loss[epoch-1]

        #check if we can top
        if patience_epochs<epoch:
            training_finished=True

    return best_validation_loss
