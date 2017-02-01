
import matplotlib; matplotlib.use("agg")


import sys
import argparse
import os
from os.path import join
import scripts.load_data.data_loader_caffe as dl
from scripts.load_data.data_loader_caffe import DataIterator
from scripts.networks import binary_classifier_caffe as bc
from scripts.util import create_run_dir, get_logger, dump_hyperparams



default_args = {'input_shape': tuple([None] + [3, 224, 224]),
                      'learning_rate': 0.00001, 
                      'dropout_p': 0.5,
                      'leakiness': 0.1,
                      'weight_decay': 0.0,
                      'num_filters': 128, 
                      'num_fc_units': 1024,
                      'num_layers': 4,
                      'momentum': 0.9,
                      'num_epochs': 20000,
                      'batch_size': 128,
                      "save_path": "None",
                      "num_tr": -1,
                      "test":False, 
                      "seed": 7,
                      "mode":"classif",
                      "exp_name": "run",
                      "load_path": "None",
                      "num_test": -1,
                      "batch_norm": False,
                    "datadir": "/global/cscratch1/sd/tkurth/atlas_dl/data_delphes",
                    "datakey": "data"
                   }



def setup_configs():
    
    # if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
    if any(["jupyter" in arg for arg in sys.argv]):
        sys.argv=sys.argv[:1]


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #make a command line argument for every flag in default args
    for k,v in default_args.iteritems():
        if type(v) is bool:
            parser.add_argument('--' + k, action='store_true', help=k)
        else:
            parser.add_argument('--' + k, type=type(v), default=v, help=k)

    args = parser.parse_args()
    

    kwargs = default_args
    kwargs.update(args.__dict__)
    
    
    kwargs = setup_res_dir(kwargs)
    
    kwargs = setup_iterators(kwargs)

    kwargs["logger"] = get_logger(kwargs['save_path'])
    
    #only classifier so far
    net = bc
        
    kwargs["net"] = net


    #kwargs["num_train"], kwargs["num_val"] = trdi.hgroup["hist"].shape[0], valdi.hgroup["hist"].shape[0]
    kwargs["logger"].info(str(kwargs))
    
    dump_hyperparams(dic=kwargs,path=kwargs["save_path"])


    return kwargs



def setup_iterators(kwargs):
    mainpath=kwargs['datadir']
    trainfiles=[mainpath+'/'+x for x in os.listdir(mainpath) if x.startswith('hep_training_')]
    validationfiles=[mainpath+'/'+x for x in os.listdir(mainpath) if x.startswith('hep_validation_')]
    testfiles=[mainpath+'/'+x for x in os.listdir(mainpath) if x.startswith('hep_test_')]
    
    loader_kwargs = dict(batch_size=kwargs["batch_size"],
                         keys=["data", "label", "normweight", "weight"])
    kwargs["loader_kwargs"] = loader_kwargs
    kwargs["trainfiles"]=trainfiles
    kwargs["validationfiles"]=validationfiles
    kwargs["testfiles"]=testfiles
    
    if not kwargs["test"]:
        #training
        trdi = DataIterator(kwargs["trainfiles"], batch_size=kwargs["batch_size"], keys=loader_kwargs['keys'])
        kwargs["tr_iterator"] = trdi
        kwargs["num_tr"] = trdi.num_events
        #validation
        valdi = DataIterator(kwargs["validationfiles"], batch_size=kwargs["batch_size"], keys=loader_kwargs['keys'])
        kwargs["val_iterator"] = valdi
        kwargs["num_val"] = valdi.num_events
        
        #shape
        kwargs["input_shape"] = tuple([None] + list(trdi.hgroup[kwargs["datakey"]].shape[1:]))
    
    else:
        #test
        tsdi = DataIterator(kwargs["testfiles"], batch_size=kwargs["batch_size"], keys=loader_kwargs['keys'])
        kwargs["test_iterator"] = tsdi
        kwargs["num_test"] = tsdi.num_events
        
        #shape
        kwargs["input_shape"] = tuple([None] + list(tsdi.hgroup[kwargs["datakey"]].shape[1:]))

    return kwargs



def setup_res_dir(kwargs):
    if kwargs["save_path"]== "None":
        kwargs["save_path"] = None

    run_dir = create_run_dir(kwargs["save_path"], name=kwargs["exp_name"])
    kwargs['save_path'] = run_dir
    return kwargs
    





