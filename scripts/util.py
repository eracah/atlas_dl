
import matplotlib; matplotlib.use("agg")


import lasagne
import theano
from theano import tensor as T
import sys
import numpy as np
import json
import pickle
import os
import logging
#enable importing of notebooks
from os import makedirs, mkdir
from os.path import join, exists
from lasagne.layers import *
# from print_n_plot import plot_ims_with_boxes, add_bbox, plot_im_with_box



def create_run_dir(results_dir=None, name=None):
  if results_dir == None:
      results_dir = './results'
  
  
  makedir_if_not_there(results_dir)
  run_num_file = os.path.join(results_dir, "run_num.txt")


  if not os.path.exists(run_num_file):
      print "making run num file...."
      f = open(run_num_file,'w')
      f.write('0')
      f.close()




  f = open(run_num_file,'r+')

  run_num = int(f.readline()) + 1

  f.seek(0)

  f.write(str(run_num))
  if name is None:
      fname = 'run'
  else:
      fname = name
  run_dir = os.path.join(results_dir,fname + str(run_num))
  os.mkdir(run_dir)
  return run_dir



def dump_hyperparams(dic, path):
    new_dic = {k:str(dic[k]) for k in dic.keys()}
    with open(path + '/hyperparams.txt', 'w') as f:
        for k,v in new_dic.iteritems():
            f.write(k + ' : ' + v + "\n")



def get_logger(run_dir):
    logger = logging.getLogger('log_train')
    if not getattr(logger, 'handler_set', None):
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler('%s/training.log'%(run_dir))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger



def makedir_if_not_there(dirname):
        if not exists(dirname):
            try:
                mkdir(dirname)
            except OSError:
                makedirs(dirname)



def iterate_minibatches(args, batchsize=128, shuffle=False):
    assert len(args[0]) == len(args[1])
    if shuffle:
        indices = np.arange(len(args[0]))
        np.random.shuffle(indices)
    if batchsize > args[0].shape[0]:
        batchsize=args[0].shape[0]
    for start_idx in range(0,len(args[0]) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [arg[excerpt] for arg in args]



def save_weights(metrics, kwargs, networks):

    def _save_weights(name,suffix=""):
        params = get_all_param_values(networks[name])
        model_dir = join(kwargs['save_path'], "models")
        makedir_if_not_there(model_dir)
        pickle.dump(params,open(join(model_dir, name + "_" + suffix + ".pkl"), "w"))


    max_metrics = ["val_acc", "val_ams", "val_sig_eff_at_cuts_bg_rej"]
    min_metrics = ["val_loss"]
    for k in max_metrics:
        if len(metrics[k]) > 1:
            if metrics[k][-1] > max(metrics[k][:-1]):
                _save_weights("net", "best_" + k)


        else:
            _save_weights("net", "best_" + k)
    for k in min_metrics:
        if len(metrics[k]) > 1:
            if metrics[k][-1] < min(metrics[k][:-1]):
                _save_weights("net", "best_" + k)





    _save_weights("net", "cur")

        



