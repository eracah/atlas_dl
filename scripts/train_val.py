
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
    def __init__(self, tr,val, kwargs, fns, networks):
        self.tr = tr
        self.val=val
        self.metrics = {}
        self.kwargs = kwargs
        self.fns = fns
        self.epoch = 0
        self.start_time = 0
        self.seed = 5
        self.networks = networks
        self.print_network(networks)

    
    def iterator(self,type_):
        data = getattr(self,type_)
        x,y, w,psr = [data[k] for k  in ["x","y", 'w', "psr"]]
        return iterate_minibatches(x,y,batchsize=self.kwargs["batch_size"], shuffle=True)
    
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
        if self.epoch > 2 :
            self.plot_learn_curve()
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
                
                if k not in metrics_tots:
                    metrics_tots[k] = 0
                metrics_tots[k] += loss_acc_dict[k]
            
   
            batches += 1
        data = getattr(self, type_)
        pred = self.fns["out"](data["x"])
        #signal confidence
        pred = pred[:,1]
        y = data["y"]
        w = data["w"]
        
        metrics_tots = {k: v/batches for k,v in metrics_tots.iteritems()}
        acc_d = {}
        
        for d in [ams(pred,y, w),
                  bg_rej_sig_eff(pred,y,w),
                  sig_eff_at(self.kwargs["sig_eff_at"], pred,y,w)]:
            for k,v in d.iteritems():
                key = type_ + "_" + k
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

        
        self.plot_roc_curve(type_)
        if type_ == "val":
            self.save_weights()

        


    def save_weights(self):
        max_metrics = ["val_acc", "val_ams", "val_sig_eff", "val_bg_rej", "val_sig_eff_at_" + str(self.kwargs["sig_eff_at"])]
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
                #print k,v
                if typ in k[:4] and "time" not in k:
                    if "acc" in k:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f} %".format(v[-1] * 100))
                    else:
                        
                        
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f}".format(v[-1]))
        
    
    def plot_roc_curve(self, type_):
        data = getattr(self, type_)
        pred = self.fns["out"](data["x"])
        #signal preds
        pred = pred[:,1]
        roc = roc_vals(pred, data["y"], data["w"])
        roc_path = join(self.kwargs['save_path'], "roc_curves")
        self.makedir_if_not_there(roc_path)
        plt.clf()
        plt.figure(1)
        plt.clf()
        plt.title('%s ROC Curve' %(type_))
        plt.plot(roc["fpr"], roc["tpr"])
        plt.legend( loc = 'center left', bbox_to_anchor = (1.0, 0.5),
           ncol=2)
        plt.savefig("%s/%s_roc_curve.png"%(roc_path,type_))
        #pass
        plt.clf()

        
        

    def plot_learn_curve(self):
        for k in self.metrics.keys():
            if "time" not in k:
                self._plot_learn_curve('_'.join(k.split("_")[1:]))
        
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
        
    
    

 
        

        





def train_val(net_cfg, kwargs, data):
    
    logger = kwargs["logger"]
    x_tr, y_tr, x_val, y_val = data
    logger.info("training set size: %i, val set size %i " %( x_tr.shape[0], x_val.shape[0]))

    tr_losses = []
    val_losses = []
    val_accs = []
    tr_accs = []
    for epoch in range(kwargs['num_epochs']):

        start = time.time() 
        tr_loss = 0
        tr_acc = 0
        for iteration, (x, y) in enumerate(iterate_minibatches(x_tr,y_tr, batchsize=kwargs['batch_size'])):
            #x = np.squeeze(x)
            loss = net_cfg['tr_fn'](x, y)
            weights = sum([np.sum(a.eval()) for a in get_all_params(net_cfg['network']) if str(a) == 'W'])
            #logger.info("weights : %6.3f" %(weights))
            #logger.info("x avg : %5.5f shape: %s : iter: %i  loss : %6.3f " % (np.mean(x), str(x.shape), iteration, loss))
            _, acc = net_cfg['val_fn'](x,y)
            logger.info("iteration % i train loss is %f"% (iteration, loss))
            logger.info("iteration % i train acc is %f"% (iteration, acc))
            tr_acc += acc
            tr_loss += loss

        train_end = time.time()
        tr_avgacc = tr_acc / (iteration + 1)
        tr_avgloss = tr_loss / (iteration + 1)


        logger.info("train time : %5.2f seconds" % (train_end - start))
        logger.info("  epoch %i of %i train loss is %f" % (epoch, kwargs["num_epochs"], tr_avgloss))
        logger.info("  epoch %i of %i train acc is %f percent" % (epoch, kwargs["num_epochs"], tr_avgacc * 100))
        tr_losses.append(tr_avgloss)
        tr_accs.append(tr_avgacc)

        val_loss = 0
        val_acc = 0
        for iteration, (xval, yval) in enumerate(iterate_minibatches(x_val,y_val, batchsize=kwargs['batch_size'])):
            #xval = np.squeeze(xval)
            loss, acc = net_cfg['val_fn'](xval, yval)
            val_loss += loss
            val_acc += acc

        val_avgloss = val_loss / (iteration + 1)
        val_avgacc = val_acc / (iteration + 1)

        logger.info("val time : %5.2f seconds" % (time.time() - train_end))
        logger.info("  epoch %i of %i val loss is %f" % (epoch, kwargs["num_epochs"], val_avgloss))
        logger.info("  epoch %i of %i val acc is %f percent" % (epoch, kwargs["num_epochs"], val_avgacc * 100))

        val_losses.append(val_avgloss)
        val_accs.append(val_avgacc)

        plot_learn_curve(tr_losses, val_losses, save_dir=kwargs["save_dir"])
        plot_learn_curve(tr_accs, val_accs, save_dir=kwargs["save_dir"], name="acc")
        pickle.dump(net_cfg['network'],open(kwargs["save_dir"] + "/model.pkl", 'w'))

    #     if epoch % 5 == 0:
    #         plot_filters(net_cfg['network'], save_dir=run_dir)
    #         for iteration, (xval, yval) in enumerate(iterate_minibatches(x_val,y_val, batchsize=batchsize)):
    #             plot_feature_maps(iteration, xval,net_cfg['network'], save_dir=run_dir)
    #             break;





