
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
    X = T.tensor4('input_var')
    Y = T.ivector('target_var')
    #network = build_layers(args)
    
    '''write loss function equation'''
    prediction = get_output(network, X)
    loss = categorical_crossentropy(prediction, Y).mean()
    weightsl2 = regularize_network_params(network, l2)
    loss += args['weight_decay'] * weightsl2
    
    '''calculate test loss (cross entropy with no regularization) and accuracy'''
    test_prediction = get_output(network, X, deterministic=True)
    test_loss = categorical_crossentropy(test_prediction, Y).mean()
    
    
    '''classification percentage: we can change this based on false postive/false negative criteria'''
    test_acc = categorical_accuracy(test_prediction,Y).mean()
    params = get_all_params(network, trainable=True)
    updates = nesterov_momentum(loss, params, learning_rate=args['learning_rate'], momentum=args['momentum'])
    
    
    '''train_fn -> takes in input,label pairs -> outputs loss '''
    train_fn = theano.function([X, Y], loss, updates=updates)
    
    
    '''val_fn -> takes in input,label pairs -> outputs non regularized loss and accuracy '''
    val_fn = theano.function([X, Y], test_loss)
    acc_fn = theano.function([X, Y], test_acc)
    out_fn = theano.function([X], test_prediction)
    return {"net":network}, {'tr': train_fn, 
                            'val': val_fn,
                            'acc': acc_fn,
                            'out': out_fn}

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
        args["logger"].info(str(layer) + str(layer.output_shape))
    print count_params(layer)
    
    return network




    



# def auc(pred,gt):
    
    



if __name__ == "__main__":
    train_fn, val_fn, network =build_network()

