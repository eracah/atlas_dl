
import matplotlib; matplotlib.use("agg")


import sys
import argparse
from os.path import join
from scripts.load_data.data_loader import DataLoader, AnomalyLoader, DataIterator
from scripts.networks import binary_classifier as bc
from scripts.networks import anom_ae as aa
from scripts.util import create_run_dir, get_logger, dump_hyperparams



def setup_configs():
    
    default_args = {'input_shape': tuple([None] + [1, 64, 64]), 
                      'learning_rate': 0.0001, 
                      'dropout_p': 0.0, 
                      'weight_decay': 0.0,
                      'num_filters': 128, 
                      'num_fc_units': 512,
                      'num_layers': 3,
                      'momentum': 0.9,
                      'num_epochs': 20000,
                      'batch_size': 128,
                      "save_path": "None",
                      "event_frac": 0.005,
                      "sig_eff_at": 0.9996,
                      "test":False, "seed": 7,
                      "mode":"classif",
                      "ae":False,
                      "tr_file":"/global/cscratch1/sd/racah/atlas_h5/train/train.h5",
                      "val_file": "/global/cscratch1/sd/racah/atlas_h5/train/val.h5",
                      "test_file": "/global/cscratch1/sd/racah/atlas_h5/test/test.h5"
                   }
    
    
    # if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
    if any(["jupyter" in arg for arg in sys.argv]):
        sys.argv=sys.argv[:1]


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #make a command line argument for every flag in default args
    for k,v in default_args.iteritems():
        parser.add_argument('--' + k, type=type(v), default=v, help=k)

    args = parser.parse_args()
    
    if args.save_path == "None":
        save_path = None
    else:
        save_path = args.save_path


    kwargs = default_args
    kwargs.update(args.__dict__)
    run_dir = create_run_dir(save_path)
    kwargs['save_path'] = run_dir
    kwargs["logger"] = get_logger(kwargs['save_path'])
    
    
    loader_kwargs = dict(groupname="all_events",
                         batch_size=128, 
                         keys=["hist", "weight", "normalized_weight", "y"])
    
    kwargs["loader_kwargs"] = loader_kwargs
    trdi = DataIterator(kwargs["tr_file"],**loader_kwargs)
    valdi = DataIterator(kwargs["val_file"],**loader_kwargs)
    kwargs["tr_iterator"] = trdi
    kwargs["val_iterator"] = valdi

    kwargs["input_shape"] = tuple([None,1] + list(trdi.hgroup["hist"].shape[1:]))
    kwargs["num_train"], kwargs["num_val"] = trdi.hgroup["hist"].shape[0], valdi.hgroup["hist"].shape[0]
    kwargs["logger"].info(str(kwargs))
    
    dump_hyperparams(dic=kwargs,path=kwargs["save_path"])
    if kwargs["ae"]:
        net = aa
    else:
        net = bc
        
    kwargs["net"] = net

    return kwargs





