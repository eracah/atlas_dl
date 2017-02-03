
import matplotlib; matplotlib.use("agg")


import sys
from scripts.metrics.objectives import bg_rej_sig_eff, calc_ams, sig_eff_at
from scripts.plotting.curve_plotter import plot_roc_curve
from scripts.load_data.data_loader_caffe import DataIterator

class MetricsProcessor(object):
    def __init__(self,kwargs):
        self.kwargs = kwargs
        self.metrics = {}
        
#         #dictionary where for each key (train,val) we have dict for all the data except for x (to save space)
#         self.data = {k: dict(y=v["y"],
#                              w =v["w"], 
#                              w_raw=v["raw_w"], 
#                              cuts=v["psr"]) 
#                      for k,v in data.iteritems()}
        
        self.cuts_metrics = self.get_cuts_metrics()
        self.metrics_tots={}
        
    def add_metrics(self, dic):
        for k in dic.keys():
            if k not in self.metrics_tots:
                self.metrics_tots[k] = 0
            self.metrics_tots[k] += dic[k]
            
    def finalize_epoch_metrics(self,num_batches):
        self.metrics_tots = {k: v / num_batches for k,v in self.metrics_tots.iteritems() }
    
    def append_metrics(self, dic, key_prefix):
        for k,v in dic.iteritems():
            key = key_prefix + k
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(v)
    
    def add_cuts_metrics(self,type_):
        self.append_metrics(self.cuts_metrics[type_], key_prefix= type_ + "_phys_cuts_")
    
    def get_data_of_type(self,type_):
        di = DataIterator(self.kwargs[type_ + "files"], keys=["psr",
                                                             "weight",
                                                             "normweight",
                                                             "label"])
        d = di.get_all()
        y, w_raw, w, cuts = [d[k] for k in ["label", "weight", "normweight", "psr"]]
        return y, w_raw, w, cuts

    def process_metrics(self, type_, pred, y, w_raw, w, cuts, time_):
        self.process_acc_metrics(type_, pred, y, w_raw, w, cuts)
        self.add_cuts_metrics(type_)
        self.append_metrics(self.metrics_tots, key_prefix=type_+"_")
        self.append_metrics({"time":time_}, key_prefix=type_ + "_")
        self.metrics_tots = {}
    
    
    def process_acc_metrics(self, type_, pred, y, w_raw, w, cuts):
        
        key_prefix=type_+"_"

        cuts_bg_rej = bg_rej_sig_eff(cuts, y, w_raw)["bg_rej"]
        
        ams = calc_ams(pred,y, w_raw)
        se_at_bg_rej = sig_eff_at(cuts_bg_rej, pred, y, w_raw, name="cuts_bg_rej")

        self.append_metrics(ams,key_prefix=key_prefix)
        self.append_metrics(se_at_bg_rej, key_prefix=key_prefix)

        
    def get_cuts_metrics(self):
        cuts_metrics = {}
        if self.kwargs["test"]:
            types = ["test"]
        else:
            types = ["train", "validation"]
        for type_ in types:
            cuts_metrics[type_] = {}
            y, w_raw, w, cuts = self.get_data_of_type(type_)

            key_prefix = type_ + "_phys_cuts_"
            cuts_ams = calc_ams(cuts,y, w_raw)
            cuts_bg_rej_sig_eff = bg_rej_sig_eff(cuts,y,w)

            cuts_metrics[type_].update(cuts_ams)
            cuts_metrics[type_].update(cuts_bg_rej_sig_eff)
        
        return cuts_metrics
    
    
    def plot_roc_curve(self, type_, pred, y, w_raw, cuts, save_path):
        plot_roc_curve(pred, y, w_raw, cuts, type_, save_path)
        





