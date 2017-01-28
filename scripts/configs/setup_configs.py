
import matplotlib; matplotlib.use("agg")


import sys
import argparse
from os.path import join
from scripts.load_data.data_loader import DataLoader, AnomalyLoader, DataIterator
from scripts.networks import binary_classifier as bc
#from scripts.networks import anom_ae as aa
from scripts.util import create_run_dir, get_logger, dump_hyperparams



default_args = {'input_shape': tuple([None] + [1, 64, 64]), 
                      'learning_rate': 0.00001, 
                      'dropout_p': 0.0, 
                      'weight_decay': 0.0,
                      'num_filters': 128, 
                      'num_fc_units': 512,
                      'num_layers': 3,
                      'momentum': 0.9,
                      'num_epochs': 20000,
                      'batch_size': 1024,
                      "save_path": "None",
                      "num_tr": -1,
                      "test":False, 
                        "seed": 7,
                      "mode":"classif",
                      "ae":False,
                      "exp_name": "run",
                      "load_path": "None",
                      "num_test": -1,
                      "tr_file":"/project/projectdirs/dasrepo/atlas_rpv_susy/hdf5/evan_curated/train.h5",
                      "val_file": "/project/projectdirs/dasrepo/atlas_rpv_susy/hdf5/evan_curated/val.h5",
                      "test_file": "/project/projectdirs/dasrepo/atlas_rpv_susy/hdf5/evan_curated/test.h5",
                      "no_batch_norm": False
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
    
    if kwargs["ae"]:
        net = aa
    else:
        net = bc
        
    kwargs["net"] = net


    #kwargs["num_train"], kwargs["num_val"] = trdi.hgroup["hist"].shape[0], valdi.hgroup["hist"].shape[0]
    kwargs["logger"].info(str(kwargs))
    
    dump_hyperparams(dic=kwargs,path=kwargs["save_path"])


    return kwargs



def setup_iterators(kwargs):
    loader_kwargs = dict(groupname="all_events",
                         batch_size=kwargs["batch_size"], 
                         keys=["hist", "weight", "normalized_weight", "y"])
    kwargs["loader_kwargs"] = loader_kwargs
    
    if not kwargs["test"]:
        trdi = DataIterator(kwargs["tr_file"],num_events=kwargs["num_tr"], **loader_kwargs)
        kwargs["tr_iterator"] = trdi

        kwargs["num_val"] = kwargs["num_tr"] if kwargs["num_tr"] == -1 else int(0.2*kwargs["num_tr"])
        valdi = DataIterator(kwargs["val_file"],num_events=kwargs["num_val"],**loader_kwargs)
        kwargs["val_iterator"] = valdi
        kwargs["input_shape"] = tuple([None,1] + list(trdi.hgroup["hist"].shape[1:]))
    
    else:
        kwargs["test_iterator"] = DataIterator(kwargs["test_file"],num_events=kwargs["num_test"],**loader_kwargs)
        kwargs["input_shape"] = tuple([None,1] + list(kwargs["test_iterator"].hgroup["hist"].shape[1:]))


    return kwargs



def setup_res_dir(kwargs):
    if kwargs["save_path"]== "None":
        kwargs["save_path"] = None

    run_dir = create_run_dir(kwargs["save_path"], name=kwargs["exp_name"])
    kwargs['save_path'] = run_dir
    return kwargs
    





