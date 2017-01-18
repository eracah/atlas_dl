
import matplotlib; matplotlib.use("agg")


import sys
import argparse
from os.path import join
from data_loader import DataLoader, AnomalyLoader
from build_network
from util import create_run_dir, get_logger, dump_hyperparams



def setup_configs():
    
    default_args = {'input_shape': tuple([None] + [1, 64, 64]), 
                      'learning_rate': 0.0001, 
                      'dropout_p': 0.0, 
                      'weight_decay': 0.0,
                      'num_filters': 64, 
                      'num_fc_units': 32,
                      'num_layers': 3,
                      'momentum': 0.9,
                      'num_epochs': 20000,
                      'batch_size': 128,
                      "save_path": "None",
                      "event_frac": 0.0005,
                      "sig_eff_at": 0.9996,
                      "test":False, "seed": 7,
                      "mode":"classif",
                      "ae":False,
                      "h5_prefix":"/home/evan/data/atlas"
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
    bg_cfg_file= [join(kwargs["h5_prefix"], "train_jetjet_JZ%i.h5"% (i)) for i in range(3,12)]
    sig_cfg_file= join(kwargs["h5_prefix"], "train_GG_RPV10_1400_850.h5")
    
    
    loader_kwargs = dict(bg_cfg_file=bg_cfg_file,
                    sig_cfg_file=sig_cfg_file,
                    events_fraction=kwargs["event_frac"], 
                    test=kwargs["test"])
    
    kwargs["loader_kwargs"] = loader_kwargs
    
    
        
    
    if kwargs["mode"] == "anomaly":
        dl = AnomalyLoader(**loader_kwargs)
    else:
        dl = DataLoader(**loader_kwargs)
    
    kwargs["data_loader"] = dl

    return kwargs



def update_configs(kwargs,tr_shape, val_shape):
    kwargs["input_shape"] = tuple([None] + list(tr_shape[1:]))
    kwargs["num_train"], kwargs["num_val"] = tr_shape[0], val_shape[0]
    kwargs["logger"].info(str(kwargs))
    
    dump_hyperparams(dic=kwargs,path=kwargs["save_path"])
    if kwargs["ae"]:
        net = caen
    else:
        net = bcc
        
    kwargs["net"] = net
    return kwargs
    





