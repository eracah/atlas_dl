
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
import copy
from theano.ifelse import ifelse



# from helper_fxns import get_best_box, get_detec_loss, get_iou, make_test_data, get_detec_acc, get_final_box
# if __name__ == "__main__":
#     from data_loader import load_classification_dataset, load_detection_dataset

def build_network(args, network):
    X = T.tensor4('X')
    #Y = T.tensor4('Y')
    thresh = 1.0
    #network = build_layers(args)
    '''write loss function equation'''
    prediction = get_output(network, X)
    loss = squared_error(prediction, X).mean()
    weightsl2 = regularize_network_params(network, l2).sum()
    loss += args['weight_decay'] * weightsl2
    
    '''calculate test loss (cross entropy with no regularization) and accuracy'''
    test_prediction = get_output(network, X, deterministic=True)
    test_loss = squared_error(test_prediction, X).sum()
    
    
    '''classification percentage: we can change this based on false postive/false negative criteria'''
    '''max reconstriuction error'''
    test_acc = test_loss 
    test_score = T.sum(squared_error(test_prediction, X), axis=(1,2,3))
    with T.autocast_float_as("float64"):
        test_score = test_score / (T.prod(X.shape[1:]))
        inds = test_score[test_score > thresh].nonzero()
        test_score = T.set_subtensor(test_score[inds], 1) 
        #test_score = ifelse(T.gt(test_score,thresh), thresh,test_score )
        test_score = 1 - test_score
    params = get_all_params(network, trainable=True)
    
    updates = adam(loss, learning_rate=args['learning_rate'], params=params)
    #updates = nesterov_momentum(loss, params, learning_rate=args['learning_rate'], momentum=args['momentum'])
    
    
    '''train_fn -> takes in input,label pairs -> outputs loss '''
    train_fn = theano.function([X], loss, updates=updates)
    
    
    '''val_fn -> takes in input,label pairs -> outputs non regularized loss and accuracy '''
    val_fn = theano.function([X], test_loss)
    acc_fn = theano.function([X], test_acc)
    out_fn = theano.function([X], test_prediction)
    score_fn = theano.function([X], test_score)
    return {"net":network}, {'tr': train_fn, 
                            'val': val_fn,
                            'acc': acc_fn,
                            'out': out_fn, 
                            "score": score_fn}

def build_layers(args):
    
    conv_kwargs = dict(num_filters=args['num_filters'], 
                       filter_size=4, pad=1,stride=2, nonlinearity=relu, W=HeNormal(gain="relu"))
    deconv_kwargs = copy.deepcopy(conv_kwargs)
    deconv_kwargs["crop"] = conv_kwargs["pad"]
    del deconv_kwargs["pad"]
    
    network = InputLayer(shape=args['input_shape'])
    for lay in range(args['num_layers']):
        network = batch_norm(Conv2DLayer(network, **conv_kwargs))
        #network = MaxPool2DLayer(network, pool_size=(2,2),stride=2)
    for lay in range(args['num_layers']):
        if lay == args['num_layers'] - 1:
            deconv_kwargs["num_filters"] = args['input_shape'][1]
            deconv_kwargs["nonlinearity"] = sigmoid
        network = Deconv2DLayer(network,**deconv_kwargs)
    #network = NonlinearityLayer(network,nonlinearity=tanh)
    
    
    for layer in get_all_layers(network):
        if "logger" in args:
            args["logger"].info(str(layer) + str(layer.output_shape))
        else:
            print str(layer) + str(layer.output_shape)
    print count_params(layer)
    
    return network


    

# def auc(pred,gt):
    
 



if __name__ == "__main__":   
    args = {'input_shape': tuple([None] + [1, 64, 64]), 
                          'learning_rate': 0.01, 
                          'dropout_p': 0, 
                          'weight_decay': 0, #0.0001, 
                          'num_filters': 10, 
                          'num_fc_units': 32,
                          'num_layers': 3,
                          'momentum': 0.9,
                          'num_epochs': 20000,
                          'batch_size': 128,
                         "save_path": "None",
                        "num_events": 100,
                        "sig_eff_at": 0.9996, "test":False, "seed": 7}

    net, fns = build_network(args, build_layers(args))
    x = np.random.random((40,1,64,64))
    

    print fns["score"](x)



# net = InputLayer(shape=(None,8,64,64))

# for i in range(3):
#     net = Conv2DLayer(net,num_filters=16, filter_size=4,pad=1, stride=2)
#     print get_output_shape(net)
# for i in range(3):
#     net = Deconv2DLayer(net,num_filters=16,filter_size=4,crop=1,stride=2)
#     print get_output_shape(net)

# get_output_shape(net)






