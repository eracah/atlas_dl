
import matplotlib; matplotlib.use("agg")


import numpy as np
import os
import lasagne
import time
import sys
from matplotlib import pyplot as plt
from matplotlib import patches



def print_train_results(epoch, num_epochs, start_time, tr_err, tr_acc):
    # Then we print the results for this epoch:
    print "Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time)
    print "\ttraining los:\t\t{:.4f}".format(tr_err)
    print "\ttraining acc:\t\t{:.4f} %".format(tr_acc * 100)


def print_val_results(val_err, val_acc):
    print "  validation loss:\t\t{:.6f}".format(val_err)
    print "  validation accuracy:\t\t{:.2f} %".format(val_acc * 100)

def plot_learn_curve(train_metric, val_metric, metric_type, save_plots, path):
        plt.figure(1 if metric_type == 'err' else 2)
        plt.clf()
        plt.title('Train/Val %s' %(metric_type))
        plt.plot(train_metric, label='train ' + metric_type)
        plt.plot(val_metric, label='val' + metric_type)
        plt.legend( bbox_to_anchor=(0.25, -0.3), loc='center left', ncol=2)
        if save_plots:
            plt.savefig("%s/%s_learning_curve.png"%(path, metric_type))
            pass
        else:
            pass

        
    





