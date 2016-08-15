
import matplotlib; matplotlib.use("agg")


import sys
import matplotlib
import argparse
from scripts.data_loader import load_data
from scripts.train_val import train
from scripts.helper_fxns import create_run_dir, dump_hyperparams, get_input_dims
from scripts.build_network import build_network



num_epochs = 11
learning_rate = 0.0001
num_events = 40
weight_decay = 0
num_filters = 128
num_fc_units = 100
dropout_p = 0
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10,
    help='number of epochs for training')

parser.add_argument('-l', '--learn_rate', default=0.0001, type=float,
    help='the learning rate for the network')

parser.add_argument('-n', '--num_events', default=40, type=int,
    help='number of total images')

args = parser.parse_args()
num_epochs = args.epochs
learning_rate = args.learn_rate
num_events = args.num_events


run_dir = create_run_dir()

dataset = load_data(num_events=50)

'''set params'''
network_kwargs = {'input_shape':(None,1,100,100), 'learning_rate': learning_rate, 'dropout_p': dropout_p, 
                  'weight_decay': weight_decay, 'num_filters': num_filters, 'num_fc_units': num_fc_units}

'''get network and train_fns'''
train_fn, val_fn, network, hyperparams = build_network(**network_kwargs)

'''save hyperparams'''
dump_hyperparams(hyperparams, path=run_dir)

'''train'''
train(dataset, network=network,train_fn=train_fn, val_fn=val_fn, num_epochs=num_epochs, save_path=run_dir)





