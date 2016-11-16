
import matplotlib; matplotlib.use("agg")


import numpy as np
import os
import lasagne
from lasagne.layers import *
import time
import sys
from matplotlib import pyplot as plt
from matplotlib import patches



import theano
import theano.tensor as T

def compile_saliency_function(net):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    """
    inp = T.tensor4('inp')
    outp = lasagne.layers.get_output(net,inp, deterministic=True)
    max_outp = T.max(outp, axis=1)
    saliency = theano.grad(max_outp.sum(), wrt=inp)
    max_class = T.argmax(outp, axis=1)
    return theano.function([inp], [saliency, max_class])

classes = ['bg', 'sig']

def show_images(img_original, saliency, max_class, title, save_dir='.'):
    # get out the first map and class from the mini-batch
    saliency = saliency[0]
    max_class = max_class[0]
    # convert saliency from BGR to RGB, and from c01 to 01c
    # plot the original image and the three saliency map variants
    im_args = dict(extent=[-3.15, 3.15, -5, 5], interpolation='none',aspect='auto', origin='low')
    plt.figure(figsize=(10, 10), facecolor='w')
    plt.suptitle("Class: " + classes[max_class] + ". Saliency: " + title)
    plt.subplot(2, 2, 1)
    plt.title('input')
    plt.imshow(np.log10(img_original).T, **im_args)
    plt.subplot(2, 2, 2)
    plt.title('abs. saliency')
    plt.imshow(np.squeeze(np.abs(saliency)).T, cmap='gray', **im_args)
    plt.subplot(2, 2, 3)
    plt.title('pos. saliency')
    plt.imshow(np.squeeze((np.maximum(0, saliency) / saliency.max())).T, **im_args)
    plt.subplot(2, 2, 4)
    plt.title('neg. saliency')
    plt.imshow(np.squeeze((np.maximum(0, -saliency) / -saliency.min())).T, **im_args)
    plt.savefig(run_dir + '/saliency_' + classes[max_class] + ".png" )
    pass



def plot_example(x):
    plt.imshow(np.log10(x).T,extent=[-3.15, 3.15, -5, 5], interpolation='none',aspect='auto', origin='low')
    plt.colorbar()

def plot_examples(x, dim, run_dir='.', name='ims'):
    plt.clf()
    assert x.shape[0] == dim**2, "not the right number examples images"
    fig, axes = plt.subplots(nrows=dim, ncols=dim, figsize=(40,40))
    for ex, ax in zip(x, axes.flat):
        im = ax.imshow(np.log10(ex).T,extent=[-3.15, 3.15, -5, 5], interpolation='none',aspect='auto', origin='low', vmin=-10, vmax=0)

    #cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    #fig.colorbar(im, cax=cax)
    #pass
    plt.savefig(run_dir + '/' + name + ".png")
        

def plot_filters(network,num_channels_to_plot = 16, save_dir='.'):
    plt.figure(figsize=(30,30))
    plt.clf()
    lay_ind = 0
    convlayers = [layer for layer in get_all_layers(network) if isinstance(layer, Conv2DLayer)]
    num_layers = len(convlayers)
    spind = 1 
    for i,layer in enumerate(convlayers):
        filters = layer.get_params()[0].eval()
        if i==0:
            filt = filters
        else:
            filt= filters[0]
        for ch_ind in range(num_channels_to_plot):
            p1 = plt.subplot(num_layers,num_channels_to_plot, spind )
            p1.imshow(filt[ch_ind], cmap="gray")
            spind = spind + 1
    
    #pass
    plt.savefig(save_dir +'/filters.png')
            

def plot_feature_maps(im, network, save_dir='.', name=""):
    plt.figure(figsize=(30,30))
    plt.clf()
    convlayers = [layer for layer in get_all_layers(network) if isinstance(layer,Conv2DLayer) or isinstance(layer,Pool2DLayer)]
    num_layers = len(convlayers)
    spind = 1 
    num_fmaps_to_plot = 16
    print im.shape
    for ch in range(num_fmaps_to_plot):
        p1 = plt.subplot(num_layers + 1,num_fmaps_to_plot, spind )
        p1.imshow(np.log10(im.T))
        spind = spind + 1
    
    # take im from 2d to 4d so the network likes it
    x=np.expand_dims(np.expand_dims(im, axis=0), axis=0)
    for layer in convlayers:
        # shape is batch_size, num_filters, x,y 
        fmaps = get_output(layer,x ).eval()
        print fmaps.shape
        for fmap_ind in range(num_fmaps_to_plot):
            p1 = plt.subplot(num_layers + 1,num_fmaps_to_plot, spind )
            p1.imshow(fmaps[0][fmap_ind].T)
            spind = spind + 1
    
    #pass
    plt.savefig(save_dir +'/fmaps_%s.png' %(name))

