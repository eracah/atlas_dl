
import matplotlib; matplotlib.use("agg")


import lasagne
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import dropout
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import rectify as relu
import theano
from theano import tensor as T
import sys
import numpy as np
#enable importing of notebooks
import inspect
# from helper_fxns import get_best_box, get_detec_loss, get_iou, make_test_data, get_detec_acc, get_final_box
# if __name__ == "__main__":
#     from data_loader import load_classification_dataset, load_detection_dataset



def build_network(learning_rate=0.001,
                  weight_decay=0.00005,
                  momentum=0.9,
                  input_shape=(None,1,100,100), 
                  weight_load_path=None,
                  nonlinearity=relu,
                  num_filters=128,
                  num_fc_units=512,
                  w_init=lasagne.init.HeUniform(),
                  dropout_p=0.5):
    
    
    input_var = T.tensor4('input_var')
    target_var = T.ivector('target_var')
    
    network = build_layers(input_var,
                           input_shape,
                           nonlinearity,
                           num_filters,
                           num_fc_units,
                           w_init,
                           dropout_p)
    
    
    # gets frame which has function keys, values in order to save hyper params
    hyperparams = get_hyperparams(inspect.currentframe())
    
    #if we are loading pretrained weights
    if weight_load_path:
        network = load_weights(weight_load_path, network)
        
    '''write loss function equation'''
    prediction = lasagne.layers.get_output(network, deterministic=False)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    loss += weight_decay * weightsl2
    
    
    '''calculate test loss (cross entropy with no regularization) and accuracy'''
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
    test_loss = test_loss.mean()
    
    '''classification percentage: we can change this based on false postive/false negative criteria'''
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var))

    '''calculate updates -> nesterov momentum sgd'''
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    #new_weight += momentum*prev_step  - leaarning_rate * (dL(cur_weight + momentum*prev_step)/dcur_weight) 
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)




    '''train_fn -> takes in input,label pairs -> outputs loss '''
    train_fn = theano.function([input_var, target_var], loss, updates=updates)


    '''val_fn -> takes in input,label pairs -> outputs non regularized loss and accuracy '''
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    return train_fn, val_fn, network, hyperparams



def build_layers(input_var,
                 input_shape,
                 nonlinearity,
                 num_filters,
                 num_fc_units,
                 w_init,
                 dropout_p):
    
    

    conv_kwargs = dict(num_filters=num_filters, filter_size=(3,3), pad=1, nonlinearity=nonlinearity, W=w_init)
    
    #eqn for reference: new_im_shape = (im_shape + 2*pad - filt_size ) / stride +1
    
    #shape: 1,100,100
    network = InputLayer(shape=input_shape, input_var=input_var)
    
    #shape: num_filters, 100,100
    network = Conv2DLayer(network, **conv_kwargs)
    
    #shape: num_filters, 50, 50
    network = MaxPool2DLayer(network, pool_size=(2,2),stride=2)
    
    #shape: num_filters, 50, 50
    network = Conv2DLayer(network, **conv_kwargs)
    
    #shape: num_filters, 25,25
    network = MaxPool2DLayer(network, pool_size=(2,2), stride=2)
    
    #shape: num_filters,24,24 (made this one different to get to an even number)
    network = Conv2DLayer(network,num_filters=num_filters, filter_size=(4,4), pad=1, 
                          nonlinearity=nonlinearity, W=w_init )
    
    #shape: num_filters, 12, 12
    network = MaxPool2DLayer(network, pool_size=(2,2), stride=2)
    
    #shape: num_fc_units
    network = dropout(network, p=dropout_p)
    network = DenseLayer(network,num_units=num_fc_units, nonlinearity=relu) 
    
    #shape: 2 (2 classes)
    network = dropout(network, p=dropout_p)
    network = DenseLayer(network, num_units=2, nonlinearity=lasagne.nonlinearities.softmax) 
    
    return network



def load_weights(file_path, network):
    '''grabs weights from an npz file'''
    with np.load(file_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    return network
    

def get_hyperparams(frame):
    #get function key, values
    args, _, _, values = inspect.getargvalues(frame)
    return values



if __name__ == "__main__":
    train_fn, val_fn, network =build_network()

