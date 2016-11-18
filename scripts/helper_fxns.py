
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
# from print_n_plot import plot_ims_with_boxes, add_bbox, plot_im_with_box



# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])



class early_stop(object):
    def __init__(self, patience=500):
        self.patience = patience   # look as this many epochs regardless
        self.patience_increase = 2  # wait this much longer when a new best is
                                      # found
        self.improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
        self.validation_frequency = self.patience // 2
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        self.best_validation_loss = np.inf

    def keep_training(self, val_loss, epoch):
        print epoch
        print val_loss
        print self.best_validation_loss
        if val_loss < self.best_validation_loss:
                #improve patience if loss improvement is good enough
                if val_loss < self.best_validation_loss *                     self.improvement_threshold:
                    self.patience = max(self.patience, epoch * self.patience_increase)

                self.best_validation_loss = val_loss
        if self.patience <= epoch:
            return False
        else:
            return True


    



def create_run_dir(results_dir=None):
    if results_dir == None:
        results_dir = './results'
    run_num_file = os.path.join(results_dir, "run_num.txt")
    if not os.path.exists(results_dir):
        print "making results dir"
        os.mkdir(results_dir)

    if not os.path.exists(run_num_file):
        print "making run num file...."
        f = open(run_num_file,'w')
        f.write('0')
        f.close()




    f = open(run_num_file,'r+')

    run_num = int(f.readline()) + 1

    f.seek(0)

    f.write(str(run_num))


    run_dir = os.path.join(results_dir,'run%i'%(run_num))
    os.mkdir(run_dir)
    return run_dir



def dump_hyperparams(dic, path):
    new_dic = {k:str(dic[k]) for k in dic.keys()}
    with open(path + '/hyperparams.json', 'w') as f:
        json.dump(new_dic, f)
#     with open(path + '/hyperparams.pkl','w') as g:
#         pickle.dump(dic, g)
    




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



def get_input_dims(tensor):
    #takes n_events by num_channels by x by y tensor
    #and returns tuple (None,num_channels, x, y )
    shape = list(tensor.shape)
    shape.pop(0)
    shape.insert(0,None)
    shape=tuple(shape)
    return shape





