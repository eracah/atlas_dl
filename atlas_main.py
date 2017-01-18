
import matplotlib; matplotlib.use("agg")


import sys
import matplotlib
import argparse
from scripts.data_loader import *
from scripts.helper_fxns import *
from scripts.print_n_plot import *
import scripts.binary_classif_convnet as bcc
import scripts.conv_ae_anom as caen
from scripts.train_val import *
import warnings
import lasagne
import theano
from theano import tensor as T
import sys
import numpy as np
import logging
import time
import pickle
import argparse
from os.path import join



def setup_kwargs():
    
    default_args = {'input_shape': tuple([None] + [1, 64, 64]), 
                      'learning_rate': 0.0001, 
                      'dropout_p': 0.0, 
                      'weight_decay': 0.0,
                      'num_filters': 10, 
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
                      "ae":False}
    
    
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

    return kwargs



if __name__ == "__main__":
    
    kwargs = setup_kwargs()
    
    h5_prefix = "/global/cscratch1/sd/racah/atlas_h5/"
    
    loader_kwargs = dict(bg_cfg_file=[join(h5_prefix, "train_jetjet_JZ%i.h5"% (i)) for i in range(3,12)],
                    sig_cfg_file=join(h5_prefix, "train_GG_RPV10_1400_850.h5"),
                    events_fraction=kwargs["event_frac"], 
                       test=kwargs["test"])
    
    if kwargs["mode"] == "anomaly":
        dl = AnomalyLoader(**loader_kwargs)
    else:
        dl = DataLoader(**loader_kwargs)
    data = dl.load_data()
    
    kwargs["input_shape"] = tuple([None] + list(data["tr"]["x"].shape[1:]))
    kwargs["num_train"], kwargs["num_val"] = data["tr"]["x"].shape[0], data["val"]["x"].shape[0]
    kwargs["logger"].info(str(kwargs))
    dump_hyperparams(dic=kwargs,path=kwargs["save_path"])
    if kwargs["ae"]:
        net = caen
    else:
        net = bcc
    networks, fns = net.build_network(kwargs, net.build_layers(kwargs))
    
    
    tv = TrainVal(data, kwargs, fns, networks)
    tv.train()

    
    
    







# h5_prefix = "/project/projectdirs/dasrepo/atlas_rpv_susy/hdf5/prod003_2016_11_14"

# a=h5py.File(join(h5_prefix,"GG_RPV10_1400_850.h5" ))

# w=a["event_10"]["weight"]

# w.value

# g.value

# # x.shape

# # #test
# # x, y, xv,yv = load_train_val(num_events=100000)

# # def test_network(network_path):
# #     x_te, y_te = load_test()

# #     net = pickle.load(open(network_path))

# #     cfg = build_network(network_kwargs,net)
# #     return cfg['val_fn'](x_te, y_te)

# # network_path = './results/run84/model.pkl'



# # net = pickle.load(open(network_path))

# # cfg = build_network(network_kwargs,net)

# # y_pred = cfg['out_fn'](xv)

# # y_pred = y_pred[0]

# # best_sig = xv[np.argmax(y_pred[:,1])]

# # best_bg = xv[np.argmin(y_pred[:,1])]

# # plot_example(np.squeeze(best_sig))

# # plot_example(np.squeeze(best_bg))

# # inds = np.argsort(y_pred[:,1], axis=0)

# # best_bgs = np.squeeze(xv[inds[:25]])

# # best_sigs = np.squeeze(xv[inds[-26:-1]])

# # plot_examples(best_bgs,5, run_dir,"best_bg")

# # plot_examples(best_sigs,5, run_dir, "best_sig")

# # plot_filters(net,save_dir=run_dir)

# # plot_feature_maps(best_bgs[0], net, run_dir, name="best_bg")

# # best_bg = np.expand_dims(np.expand_dims(best_bgs[0], axis=0),axis=0)
# # best_sig = np.expand_dims(np.expand_dims(best_sigs[-1], axis=0),axis=0)
# # saliency_fn = compile_saliency_function(net)
# # saliency, max_class = saliency_fn(best_sig)
# # #np.squeeze(np.abs(saliency)).shape
# # show_images(best_sigs[-1], saliency, max_class, "default gradient", save_dir=run_dir)




