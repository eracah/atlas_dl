
import matplotlib; matplotlib.use("agg")


import sys
import matplotlib
import argparse
from scripts.data_loader import *
from scripts.helper_fxns import *
from scripts.print_n_plot import *
from scripts.build_network import *
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



import numpy as np
import lasagne
import time
import sys
from matplotlib import pyplot as plt
import json
import pickle
from matplotlib import patches
from os.path import join, exists
from os import mkdir, makedirs
# from helper_fxns import early_stop
# from print_n_plot import *
# from build_network import build_network
import logging
#from data_loader import load_data

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    if batchsize > inputs.shape[0]:
        batchsize=inputs.shape[0]
    for start_idx in range(0,len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


class TrainVal(object):
    def __init__(self, data, kwargs, fns, networks):
        self.data = data
        self.metrics = {}
        self.kwargs = kwargs
        self.fns = fns
        self.epoch = 0
        self.start_time = 0
        self.seed = 5
        self.networks = networks
        self.print_network(networks)

    
    def iterator(self,type_):
        x,y = self.data[type_+"_x"], self.data[type_+"_y"]
        return iterate_minibatches(x,y, batchsize=self.kwargs["batch_size"], shuffle=True)
    
    def print_network(self, networks):
        for net in networks.values():
            self._print_network(net)
            
            
    def _print_network(self, network):
        self.kwargs['logger'].info("\n")
        for layer in get_all_layers(network):
            self.kwargs['logger'].info(str(layer) +' : ' + str(layer.output_shape))
        self.kwargs['logger'].info(str(count_params(layer)))
        self.kwargs['logger'].info("\n")
    
    def do_one_epoch(self):
        self._do_one_epoch(type_="tr")
        self._do_one_epoch(type_="val")
        self.print_results()
        self.epoch += 1
    
    def _do_one_epoch(self, type_="tr"):
        print "beginning epoch %i" % (self.epoch)
        start_time = time.time()
        metrics_tots = {}
        batches = 0
        for x,y in self.iterator(type_):
            loss = self.fns[type_](x,y)
            
            acc = self.fns["acc"](x,y)
            
            loss_acc_dict = dict(loss=loss, acc=acc)
            
            
            for k in loss_acc_dict.keys():
                key = type_ + "_" + k
                if key not in metrics_tots:
                    metrics_tots[key] = 0
                metrics_tots[key] += loss_acc_dict[k]
            
   
            batches += 1
        assert batches > 0
        for k,v in metrics_tots.iteritems():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v / float(batches))

        time_key = type_ + "_time"
        if time_key not in self.metrics:
            self.metrics[time_key] = []
        self.metrics[time_key].append(time.time() - start_time)

        if type_ == "val":
            self.save_weights()

        


    def save_weights(self):
        best_metrics = ["val_loss", "val_acc"]
        for k in best_metrics:
            if len(self.metrics[k]) > 1:
                if self.metrics[k][-1] > max(self.metrics[k][:-1]):
                    self._save_weights("net", "best_" + k)
            else:
                self._save_weights("net", "best_" + k)





        self._save_weights("net", "cur") 
        
    def _save_weights(self,name,suffix=""):
        params = get_all_param_values(self.networks[name])
        model_dir = join(self.kwargs['save_path'], "models")
        self.makedir_if_not_there(model_dir)
        pickle.dump(params,open(join(model_dir, name + "_" + suffix + ".pkl"), "w"))
        
    def makedir_if_not_there(self, dirname):
        if not exists(dirname):
            try:
                mkdir(dirname)
            except OSError:
                makedirs(dirname)
        
        
    def print_results(self):
        self.kwargs['logger'].info("Epoch {} of {} took {:.3f}s".format(self.epoch + 1, self.kwargs['num_epochs'],
                                                                  self.metrics["tr_time"][-1]))
        for typ in ["tr", "val"]:
            if typ == "val":
                self.kwargs['logger'].info("\tValidation took {:.3f}s".format(self.metrics["val_time"][-1]))
            for k,v in self.metrics.iteritems():
                if typ in k[:4] and "time" not in k:
                    if "acc" in k:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f} %".format(v[-1] * 100))
                    else:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f}".format(v[-1]))
        
        

    def plot_learn_curve(self):
        for k in self.metrics.keys():
            if "time" not in k:
                self._plot_learn_curve(k.split("_")[1])
        
    def _plot_learn_curve(self,type_):
        plt.clf()
        plt.figure(1)
        plt.clf()
        plt.title('Train/Val %s' %(type_))
        plt.plot(self.metrics['tr_' + type_], label='train ' + type_)
        plt.plot(self.metrics['val_' + type_], label='val ' + type_)
        plt.legend( loc = 'center left', bbox_to_anchor = (1.0, 0.5),
           ncol=2)

        curves_path = join(self.kwargs['save_path'], "learn_curves")
        self.makedir_if_not_there(curves_path)
        plt.savefig("%s/%s_learning_curve.png"%(curves_path,type_))
        pass
        plt.clf()
        
    
    

 
        

        



def setup_kwargs():
    
    default_args = {'input_shape':tuple([None] + [1, 50, 50]), 
                      'learning_rate': 0.1, 
                      'dropout_p': 0, 
                      'weight_decay': 0.0001, 
                      'num_filters': 10, 
                      'num_fc_units': 32,
                      'num_layers': 4,
                      'momentum': 0.9,
                      'num_epochs': 10000,
                      'batch_size': 128,
                     "save_path": "None"}
    
    
    # if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
    if any(["jupyter" in arg for arg in sys.argv]):
        sys.argv=sys.argv[:1]
        #default_args.update({"lambda_ae":0,"yolo_load_path":"./results/run289/models/yolo.pkl", "get_ims":True, "conf_thresh": 0.5 })



    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    
    h5_prefix = "/project/projectdirs/dasrepo/atlas_rpv_susy/hdf5/prod002_2016_11_10"
    
    dl = DataLoader(bg_cfg_file=join(h5_prefix, "jetjet_JZ4.h5"), 
                    sig_cfg_file=join(h5_prefix, "GG_RPV10_1400_850.h5"),
                   num_events=20, type_="hdf5",use_premade=True)
    x,y,xv,yv = dl.load_data()
    data = dict(tr_x=x,tr_y=y, val_x=xv, val_y=yv)

    kwargs["logger"].info(str(kwargs))
    networks, fns = build_network(kwargs, build_layers(kwargs))
    tv = TrainVal(data, kwargs, fns, networks)
    for epoch in range(kwargs["num_epochs"]):
        tv.do_one_epoch()
    
    
    







x.shape

#test
x, y, xv,yv = load_train_val(num_events=100000)

def test_network(network_path):
    x_te, y_te = load_test()

    net = pickle.load(open(network_path))

    cfg = build_network(network_kwargs,net)
    return cfg['val_fn'](x_te, y_te)

network_path = './results/run84/model.pkl'



net = pickle.load(open(network_path))

cfg = build_network(network_kwargs,net)

y_pred = cfg['out_fn'](xv)

y_pred = y_pred[0]

best_sig = xv[np.argmax(y_pred[:,1])]

best_bg = xv[np.argmin(y_pred[:,1])]

plot_example(np.squeeze(best_sig))

plot_example(np.squeeze(best_bg))

inds = np.argsort(y_pred[:,1], axis=0)

best_bgs = np.squeeze(xv[inds[:25]])

best_sigs = np.squeeze(xv[inds[-26:-1]])

plot_examples(best_bgs,5, run_dir,"best_bg")

plot_examples(best_sigs,5, run_dir, "best_sig")

plot_filters(net,save_dir=run_dir)

plot_feature_maps(best_bgs[0], net, run_dir, name="best_bg")

best_bg = np.expand_dims(np.expand_dims(best_bgs[0], axis=0),axis=0)
best_sig = np.expand_dims(np.expand_dims(best_sigs[-1], axis=0),axis=0)
saliency_fn = compile_saliency_function(net)
saliency, max_class = saliency_fn(best_sig)
#np.squeeze(np.abs(saliency)).shape
show_images(best_sigs[-1], saliency, max_class, "default gradient", save_dir=run_dir)










