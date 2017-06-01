
import matplotlib; matplotlib.use("agg")


import sys
import pickle
from scripts.metrics.objectives import *
from os.path import join, exists
from os import makedirs, mkdir
from scripts.plotting.curve_plotter import plot_roc_curve, plot_learn_curve
from scripts.util import makedir_if_not_there, iterate_minibatches, save_weights
from scripts.metrics.metrics_processor import MetricsProcessor
from scripts.printing.print_utils import *
import time
from lasagne.layers import *
import numpy as np

class TrainVal(object):
    def __init__(self, kwargs, fns, networks):
        self.kwargs = kwargs
        self.fns = fns
        self.epoch = 0
        self.networks = networks
        print_network(networks, self.kwargs)
        self.mp = MetricsProcessor(kwargs)
        self.epoch_time = 0

        
    def train(self):
        for epoch in range(self.kwargs["num_epochs"]):
            self.train_one_epoch()
            
    def test(self):
        self._do_one_epoch(type_="test")
        print_results(self.kwargs, self.epoch, self.mp.metrics)
    
    def iterator(self,type_):
        for item in self.kwargs[type_ +"_iterator"]:
            yield item
    
    def train_one_epoch(self):
        self._do_one_epoch(type_="tr")
        self._do_one_epoch(type_="val")
        save_weights(self.mp.metrics, self.kwargs, self.networks)
        plot_learn_curve(self.mp.metrics, self.kwargs["save_path"])
        print_results(self.kwargs, self.epoch, self.mp.metrics)
        self.epoch += 1
        
    def _do_one_epoch(self, type_="tr"):
        print("beginning epoch %i %s" % (self.epoch, type_))
        self.do_learn_loop(type_)
        self.postprocess(type_)
    
    def do_learn_loop(self,type_):
                
        start_time = time.time()
        batches = 0
        for minibatch in self.iterator(type_):
            x,y,w = [minibatch[k] for k in ["hist", "y", "normalized_weight"]]
            loss = self.fns[type_](x,y,w)
            acc = self.fns["acc"](x,y,w)
            self.mp.add_metrics(dict(loss=loss, acc=acc))
            batches += 1
        self.epoch_time = time.time() - start_time
                
        self.mp.finalize_epoch_metrics(batches)
    
    def fprop(self,type_):
        pred = []
        for minibatch in self.iterator(type_):
            x = minibatch["hist"]
            p = self.fns["score"](x)
            pred.extend(p)
        return pred
    
    def postprocess(self,type_):
        pred = self.fprop(type_)
        self.mp.process_metrics(type_, pred, self.epoch_time)
        self.mp.plot_roc_curve(type_, pred, self.kwargs["save_path"])





