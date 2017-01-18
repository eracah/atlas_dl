
import matplotlib; matplotlib.use("agg")


import numpy as np
import lasagne
from lasagne.layers import *
import time
import sys
from matplotlib import pyplot as plt
import json
import pickle
from matplotlib import patches
import logging
from objectives import *
from os.path import join, exists
from os import makedirs, mkdir
from curve_plotter import plot_roc_curve, plot_learn_curve

from util import makedir_if_not_there, iterate_minibatches
#from data_loader import load_data



class TrainVal(object):
    def __init__(self, data, kwargs, fns, networks):
        self.data=data
        self.metrics = {}
        self.kwargs = kwargs
        self.fns = fns
        self.epoch = 0
        self.start_time = 0
        self.seed = 5
        self.networks = networks
        self.print_network(networks)

    
    def iterator(self,type_):
        data = self.data[type_]
        x, y, w, psr = [data[k] for k  in ["x","y", 'w', "psr"]]
        return iterate_minibatches([x,y,w],batchsize=self.kwargs["batch_size"], shuffle=True)
    
    def print_network(self, networks):
        for net in networks.values():
            self._print_network(net)
            
            
    def _print_network(self, network):
        self.kwargs['logger'].info("\n")
        for layer in get_all_layers(network):
            self.kwargs['logger'].info(str(layer) +' : ' + str(layer.output_shape))
        self.kwargs['logger'].info(str(count_params(layer)))
        self.kwargs['logger'].info("\n")
    
    def train_one_epoch(self):
        self._do_one_epoch(type_="tr")
        self._do_one_epoch(type_="val")
        
        plot_learn_curve(self.metrics, self.kwargs["save_path"])
        self.print_results()
        self.epoch += 1
    
    def train(self):
        for epoch in range(self.kwargs["num_epochs"]):
            self.train_one_epoch()
            
    def test(self):
        self._do_one_epoch(type_="test")
        self.print_results()
        
    def _do_one_epoch(self, type_="tr"):
        print "beginning epoch %i" % (self.epoch)
        start_time = time.time()
        metrics_tots = {}
        batches = 0
        for x,y,w in self.iterator(type_):
            w=np.squeeze(w)
            if self.kwargs["ae"]:
                #TODO make weighted loss for ae
                args = [x]
            else:
                args = [x,y,w]
            
            loss = self.fns[type_](*args)
            acc = self.fns["acc"](*args)
            
            

            loss_acc_dict = dict(loss=loss, acc=acc)

            
            for k in loss_acc_dict.keys():
                
                if k not in metrics_tots:
                    metrics_tots[k] = 0
                metrics_tots[k] += loss_acc_dict[k]
            
   
            batches += 1
        data = self.data[type_]
        pred = self.fns["score"](data["x"])
        #signal confidence
        y = data["y"]
        w = data["w"]
        w_raw = data["raw_w"]
        
        
        cuts = data["psr"]
        
        metrics_tots = {k: v/ batches for k,v in metrics_tots.iteritems()}
        acc_d = {}
        

        cuts_bg_rej = bg_rej_sig_eff(cuts,y,w)["bg_rej"]
        for d in [ams(pred,y, w_raw, self.kwargs["event_frac"]),
                  bg_rej_sig_eff(pred,y,w),
                  sig_eff_at(cuts_bg_rej, pred,y,w,name="cuts_bg_rej"),
                  sig_eff_at(0.9999, pred,y,w)]:

            for k,v in d.iteritems():
                key = type_ + "_" + k
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(v)

        for d in [ams(cuts,y, w_raw,self.kwargs["event_frac"] ),
                  bg_rej_sig_eff(cuts,y,w)]:

            for k,v in d.iteritems():
                key = type_ + "_phys_cuts_" + k
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(v)

            

        assert batches > 0
        for k,v in metrics_tots.iteritems():
            key = type_ + "_" + k
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(v)

        time_key = type_ + "_time"
        if time_key not in self.metrics:
            self.metrics[time_key] = []
        self.metrics[time_key].append(time.time() - start_time)

        data = self.data[type_]
        pred = self.fns["score"](data["x"])
        y = data["y"]
        w = data["w"]
        cuts = data["psr"]
        plot_roc_curve(pred, y,w, cuts,type_, self.kwargs["save_path"])
        if type_ == "val":
            self.save_weights()

        


    def save_weights(self):
        max_metrics = ["val_acc", "val_ams", "val_sig_eff", "val_bg_rej", "val_sig_eff_at_cuts_bg_rej"]
        min_metrics = ["val_loss"]
        for k in max_metrics:
            if len(self.metrics[k]) > 1:
                if self.metrics[k][-1] > max(self.metrics[k][:-1]):
                    self._save_weights("net", "best_" + k)
        
        
            else:
                self._save_weights("net", "best_" + k)
        for k in min_metrics:
            if len(self.metrics[k]) > 1:
                if self.metrics[k][-1] < min(self.metrics[k][:-1]):
                    self._save_weights("net", "best_" + k)





        self._save_weights("net", "cur")

        


        
    def _save_weights(self,name,suffix=""):
        params = get_all_param_values(self.networks[name])
        model_dir = join(self.kwargs['save_path'], "models")
        makedir_if_not_there(model_dir)
        pickle.dump(params,open(join(model_dir, name + "_" + suffix + ".pkl"), "w"))

        
     
    
    def print_results(self):
        self.kwargs['logger'].info("Epoch {} of {} took {:.3f}s".format(self.epoch + 1, self.kwargs['num_epochs'],
                                                                  self.metrics["tr_time"][-1]))
        for typ in ["tr", "val"]:
            if typ == "val":
                self.kwargs['logger'].info("\tValidation took {:.3f}s".format(self.metrics["val_time"][-1]))
            for k,v in self.metrics.iteritems():
                #print k,v
                val = v[-1][0] if isinstance(v[-1], list) or isinstance(v[-1], np.ndarray)  else v[-1]
                if typ in k[:4] and "time" not in k:
                    if "ams" not in k and "loss" not in k:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f} %".format(val * 100))
                            
                    else:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f}".format(val))
        
    

        
    
    

 
        

        





