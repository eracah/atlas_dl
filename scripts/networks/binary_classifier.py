
import matplotlib; matplotlib.use("agg")


import lasagne
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.regularization import regularize_network_params, l2
from lasagne.updates import *
from lasagne.init import *
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import *
import theano
from theano import tensor as T
import sys
import numpy as np
#enable importing of notebooks
import inspect



# from helper_fxns import get_best_box, get_detec_loss, get_iou, make_test_data, get_detec_acc, get_final_box
# if __name__ == "__main__":
#     from data_loader import load_classification_dataset, load_detection_dataset

def build_network(args, network):
    X = T.tensor4('X')
    Y = T.ivector('Y')
    
    #physics weights
    W = T.dvector('W')
    
    #make sum to 1
    #w = W / T.sum(W)
    #network = build_layers(args)
    
    '''write loss function equation'''
    prediction = get_output(network, X)
    loss = categorical_crossentropy(prediction, Y)
    
    #multiply by weights
    loss =  T.dot(loss.T,W)
    weightsl2 = regularize_network_params(network, l2)
    loss += args['weight_decay'] * weightsl2
    
    '''calculate test loss (cross entropy with no regularization) and accuracy'''
    test_prediction = get_output(network, X, deterministic=True)
    test_loss = categorical_crossentropy(test_prediction, Y)
    test_loss = T.dot(test_loss.T,W)
    
    
    '''classification percentage: we can change this based on false postive/false negative criteria'''
    test_acc = categorical_accuracy(test_prediction,Y)
    test_acc = T.dot(test_acc.T,W) / T.sum(W)
    params = get_all_params(network, trainable=True)
    
    updates = adam(loss, learning_rate=args['learning_rate'], params=params)
    #updates = nesterov_momentum(loss, params, learning_rate=args['learning_rate'], momentum=args['momentum'])
    
    
    '''train_fn -> takes in input,label pairs -> outputs loss '''
    train_fn = theano.function([X, Y, W], loss, updates=updates)
    
    
    '''val_fn -> takes in input,label pairs -> outputs non regularized loss and accuracy '''
    val_fn = theano.function([X, Y, W], test_loss)
    acc_fn = theano.function([X, Y, W], test_acc)
    out_fn = theano.function([X], test_prediction)
    score_fn = theano.function([X], test_prediction[:,1].T)
    return {"net":network}, {'tr': train_fn, 
                            'val': val_fn,
                            'acc': acc_fn,
                            'out': out_fn, "score":score_fn}

def build_layers(args):
    
    conv_kwargs = dict(num_filters=args['num_filters'], filter_size=3, pad=1, nonlinearity=relu, W=HeNormal(gain="relu"))
    network = InputLayer(shape=args['input_shape'])
    for lay in range(args['num_layers']):
        network = batch_norm(Conv2DLayer(network, **conv_kwargs))
        network = MaxPool2DLayer(network, pool_size=(2,2),stride=2)
    network = dropout(network, p=args['dropout_p'])
    network = DenseLayer(network,num_units=args['num_fc_units'], nonlinearity=relu) 
    network = dropout(network, p=args['dropout_p'])
    network = DenseLayer(network, num_units=2, nonlinearity=softmax)
    
    for layer in get_all_layers(network):
        if "logger" in args:
            args["logger"].info(str(layer) + str(layer.output_shape))
    print count_params(layer)
    
    return network


    

# def auc(pred,gt):
    
 





if __name__ == "__main__":
 inp_d = {'input_shape': tuple([None] + [1, 64, 64]), 
                   'learning_rate': 0.01, 
                   'dropout_p': 0.5, 
                   'weight_decay': 0, #0.0001, 
                   'num_filters': 10, 
                   'num_fc_units': 32,
                   'num_layers': 3,
                   'momentum': 0.9,
                   'num_epochs': 20000,
                   'batch_size': 128,
                   "save_path": "None",
                   "num_events": 200,
                   "sig_eff_at": 0.9996,
                   "test":False, "seed": 7,
                   "mode":"classif",
                   "ae":False}
 net, fns = build_network(inp_d, build_layers(inp_d))



# conv_kwargs = dict(num_filters=5, filter_size=3, pad=1, nonlinearity=relu, W=HeNormal(gain="relu"))
# network = InputLayer(shape=(None,3,50,50))
# for lay in range(3):
#     network = batch_norm(Conv2DLayer(network, **conv_kwargs))
#     network = MaxPool2DLayer(network, pool_size=(2,2),stride=2)
# network = DenseLayer(network,num_units=10, nonlinearity=relu) 
# network = DenseLayer(network, num_units=3, nonlinearity=softmax)

# X = T.tensor4('X')
# Y = T.ivector('Y')
# W = T.vector('W')

# W = W / T.sum(W)
# #network = build_layers(args)

# '''write loss function equation'''
# prediction = get_output(network, X)
# loss = categorical_crossentropy(prediction, Y)
# newloss = T.dot(loss.T,W) / W.shape[0]
# loss = loss.mean()
# #loss = loss * W
# #loss = loss / W.shape[0]
# val_fn = theano.function([X, Y, W], [loss, newloss, W])

# x = np.random.random((64,3,50,50))

# y = np.random.randint(0,3,size=(64,))

# y=y.astype("int32")

# w = 10**3 * np.random.random((64,))

# val_fn(x,y,w)






