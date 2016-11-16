
import matplotlib; matplotlib.use("agg")


import numpy as np
import lasagne
import time
import sys
from matplotlib import pyplot as plt
import json
import pickle
from matplotlib import patches
from helper_fxns import early_stop
from print_n_plot import *
from build_network import build_network
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





