
import matplotlib; matplotlib.use("agg")


import sys
import matplotlib
import argparse
from scripts.data_loader import load_data
from scripts.helper_fxns import create_run_dir
from scripts.print_n_plot import *
import warnings
import lasagne
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.regularization import regularize_network_params, l2
from lasagne.updates import *
from lasagne.init import HeNormal
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import *
import theano
from theano import tensor as T
import sys
import numpy as np
import logging
import time
import pickle
import argparse



# if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]


parser = argparse.ArgumentParser()

parser.add_argument('-l', '--learn_rate', default=0.01, type=float,
    help='the learning rate for the network')

parser.add_argument('-n', '--num_ims', default=2000, type=int,
    help='number of total images')

parser.add_argument('-f', '--num_filters', default=128, type=int,
    help='number of filters in each conv layer')

parser.add_argument( '--fc', default=1024, type=int,
    help='number of fully connected units')


args = parser.parse_args()



def build_network(args):
    X = T.tensor4('input_var')
    Y = T.ivector('target_var')
    network = build_layers(args)
    '''write loss function equation'''
    prediction = get_output(network, X)
    loss = categorical_crossentropy(prediction, Y).mean()
    weightsl2 = regularize_network_params(network, l2)
    loss += args['weight_decay'] * weightsl2
    
    '''calculate test loss (cross entropy with no regularization) and accuracy'''
    test_prediction = get_output(network, X, deterministic=True)
    test_loss = categorical_crossentropy(test_prediction, Y).mean()
    
    '''classification percentage: we can change this based on false postive/false negative criteria'''
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), Y))
    params = get_all_params(network, trainable=True)
    #new_weight += momentum*prev_step  - leaarning_rate * (dL(cur_weight + momentum*prev_step)/dcur_weight) 
    updates = nesterov_momentum(loss, params, learning_rate=args['learning_rate'], momentum=args['momentum'])
    '''train_fn -> takes in input,label pairs -> outputs loss '''
    train_fn = theano.function([X, Y], loss, updates=updates)
    '''val_fn -> takes in input,label pairs -> outputs non regularized loss and accuracy '''
    val_fn = theano.function([X, Y], [test_loss, test_acc])

    return {
            'tr_fn': train_fn, 
            'val_fn': val_fn, 
            'network': network
            
            }

def build_layers(args):
    
    conv_kwargs = dict(num_filters=args['num_filters'], filter_size=3, pad=1, nonlinearity=relu, W=HeNormal())
    network = InputLayer(shape=args['input_shape'])
    for lay in range(args['num_layers']):
        network = batch_norm(Conv2DLayer(network, **conv_kwargs))
        network = MaxPool2DLayer(network, pool_size=(2,2),stride=2)
    network = dropout(network, p=args['dropout_p'])
    network = DenseLayer(network,num_units=args['num_fc_units'], nonlinearity=relu) 
    network = dropout(network, p=args['dropout_p'])
    network = DenseLayer(network, num_units=2, nonlinearity=softmax)
    
    for layer in get_all_layers(network):
        logger.info(str(layer) + str(layer.output_shape))
    print count_params(layer)
    
    return network



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



run_dir = create_run_dir()



x, y = load_data(num_events=args.num_ims, bin_size=0.1)




inds = np.arange(x.shape[0])

#shuffle data
rng = np.random.RandomState(7)
rng.shuffle(inds)

#split train, val, test
tr_inds = inds[:int(0.8*len(inds))] 
val_inds = inds[int(0.8*len(inds)):]

x_tr, y_tr, x_val, y_val = x[tr_inds], y[tr_inds], x[val_inds], y[val_inds]


'''a type of sparse preprocessing, which scales everything between -1 and 1 without losing sparsity'''
#only calculate the statistic using training set
max_abs=np.abs(x_tr).max(axis=(0,1,2,3))

#then scale all sets
x_tr /= max_abs
x_val /= max_abs





num_epochs = 5000
batchsize = 128
try:
    print logger
except:
    logger = logging.getLogger('log_train')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('%s/training.log'%(run_dir))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    
logger.info("training set size: %i, val set size %i " %( x_tr.shape[0], x_val.shape[0]))
    
'''set params'''
                    # (None,1,62,100) -> none b/c batch size could vary
network_kwargs = {'input_shape':tuple([None] + [x.shape[i] for i in [1,2,3]]), 
                  'learning_rate': args.learn_rate, 
                  'dropout_p': 0, 
                  'weight_decay': 0.0001, 
                  'num_filters': args.num_filters, 
                  'num_fc_units': args.fc,
                  'num_layers': 4,
                  'momentum': 0.9}

logger.info(str(network_kwargs))

'''get network and train_fns'''
net_cfg = build_network(network_kwargs)



tr_losses = []
val_losses = []
val_accs = []
tr_accs = []
for epoch in range(num_epochs):

    start = time.time() 
    tr_loss = 0
    tr_acc = 0
    for iteration, (x, y) in enumerate(iterate_minibatches(x_tr,y_tr, batchsize=batchsize)):
        #x = np.squeeze(x)
        loss = net_cfg['tr_fn'](x, y)
        weights = sum([np.sum(a.eval()) for a in get_all_params(net_cfg['network']) if str(a) == 'W'])
        logger.info("weights : %6.3f" %(weights))
        logger.info("x avg : %5.5f shape: %s : iter: %i  loss : %6.3f " % (np.mean(x), str(x.shape), iteration, loss))
        _, acc = net_cfg['val_fn'](x,y)
        tr_acc += acc
        tr_loss += loss
    
    train_end = time.time()
    tr_avgacc = tr_acc / (iteration + 1)
    tr_avgloss = tr_loss / (iteration + 1)
    
    
    logger.info("train time : %5.2f seconds" % (train_end - start))
    logger.info("  epoch %i of %i train loss is %f" % (epoch, num_epochs, tr_avgloss))
    logger.info("  epoch %i of %i train acc is %f percent" % (epoch, num_epochs, tr_avgacc * 100))
    tr_losses.append(tr_avgloss)
    tr_accs.append(tr_avgacc)
    
    val_loss = 0
    val_acc = 0
    for iteration, (xval, yval) in enumerate(iterate_minibatches(x_val,y_val, batchsize=batchsize)):
        #xval = np.squeeze(xval)
        loss, acc = net_cfg['val_fn'](xval, yval)
        val_loss += loss
        val_acc += acc
    
    val_avgloss = val_loss / (iteration + 1)
    val_avgacc = val_acc / (iteration + 1)
    
    logger.info("val time : %5.2f seconds" % (time.time() - train_end))
    logger.info("  epoch %i of %i val loss is %f" % (epoch, num_epochs, val_avgloss))
    logger.info("  epoch %i of %i val acc is %f percent" % (epoch, num_epochs, val_avgacc * 100))
    
    val_losses.append(val_avgloss)
    val_accs.append(val_avgacc)
    
    plot_learn_curve(tr_losses, val_losses, save_dir=run_dir)
    plot_learn_curve(tr_accs, val_accs, save_dir=run_dir, name="acc")
    pickle.dump(net_cfg['network'],open(run_dir + "/model.pkl", 'w'))
    
#     if epoch % 5 == 0:
#         plot_filters(net_cfg['network'], save_dir=run_dir)
#         for iteration, (xval, yval) in enumerate(iterate_minibatches(x_val,y_val, batchsize=batchsize)):
#             plot_feature_maps(iteration, xval,net_cfg['network'], save_dir=run_dir)
#             break;

































