
import matplotlib; matplotlib.use("agg")


import numpy as np
import os
import lasagne
from lasagne.layers import *
import time
import sys
from matplotlib import pyplot as plt
from matplotlib import patches



# def print_train_results(epoch, num_epochs, start_time, tr_err, tr_acc):
#     # Then we print the results for this epoch:
#     print "Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time)
#     print "\ttraining los:\t\t{:.4f}".format(tr_err)
#     print "\ttraining acc:\t\t{:.4f} %".format(tr_acc * 100)


# def print_val_results(val_err, val_acc):
#     print "  validation loss:\t\t{:.6f}".format(val_err)
#     print "  validation accuracy:\t\t{:.2f} %".format(val_acc * 100)

    
    
    
    
# def plot_learn_curve(train_metric, val_metric, metric_type, save_plots, path):
#         plt.figure(1 if metric_type == 'err' else 2)
#         plt.clf()
#         plt.title('Train/Val %s' %(metric_type))
#         plt.plot(train_metric, label='train ' + metric_type)
#         plt.plot(val_metric, label='val' + metric_type)
#         plt.legend( bbox_to_anchor=(0.25, -0.3), loc='center left', ncol=2)
#         if save_plots:
#             plt.savefig("%s/%s_learning_curve.png"%(path, metric_type))
#             pass
#         else:
#             pass

        
    



def plot_learn_curve(tr_losses, val_losses, save_dir='.', name="loss"):
    plt.clf()
    plt.plot(tr_losses)
    plt.plot(val_losses)
    plt.savefig(save_dir + '/%s_learn_curve.png'%(name))
    plt.clf()
    
# def plot_clusters(i,x,y,net_cfg, save_dir='.'):
#     x = np.squeeze(x)
#     hid_L = net_cfg['h_fn'](x)
#     ts = TSNE().fit_transform(hid_L)
#     plt.clf()
#     plt.scatter(ts[:,0], ts[:,1], c=y)
#     plt.savefig(save_dir + '/cluster_%i.png'%(i))
#     plt.clf()

# def plot_recs(i,x,out_fn, save_dir='.'):
#     ind = np.random.randint(0,x.shape[0], size=(1,))
#     x=np.squeeze(x)
#     #print x.shape
#     im = x[ind]
#     #print im.shape
#     rec = out_fn(im)
#     ch=1
#     plt.figure(figsize=(30,30))
#     plt.clf()
#     for (p_im, p_rec) in zip(im[0],rec[0]):
#         p1 = plt.subplot(im.shape[1],2, ch )
#         p2 = plt.subplot(im.shape[1],2, ch + 1)
#         p1.imshow(p_im)
#         p2.imshow(p_rec)
#         ch = ch+2
#     #pass
#     plt.savefig(save_dir +'/recs_%i' %(i))

def plot_filters(network, save_dir='.'):
    plt.figure(figsize=(30,30))
    plt.clf()
    lay_ind = 0
    num_channels_to_plot = 16
    convlayers = [layer for layer in get_all_layers(network) if isinstance(layer, Conv2DLayer)]
    num_layers = len(convlayers)
    spind = 1 
    for layer in convlayers:
        filters = layer.get_params()[0].eval()
        #pick a random filter
        filt = filters #filters[np.random.randint(0,filters.shape[0])]
        for ch_ind in range(num_channels_to_plot):
            p1 = plt.subplot(num_layers,num_channels_to_plot, spind )
            p1.imshow(filt[ch_ind], cmap="gray")
            spind = spind + 1
    
    #pass
    plt.savefig(save_dir +'/filters.png')
            
        
def plot_feature_maps(i, x, network, save_dir='.'):
    plt.figure(figsize=(30,30))
    plt.clf()
    ind = np.random.randint(0,x.shape[0])


    im = x[ind]
    convlayers = [layer for layer in get_all_layers(network) if not isinstance(layer,DenseLayer) or  not isinstance(layer,InputLayer)]
    num_layers = len(convlayers)
    spind = 1 
    num_fmaps_to_plot = 16
    print im.shape
    for ch in range(im.shape[0]):
        p1 = plt.subplot(num_layers + 1,num_fmaps_to_plot, spind )
        p1.imshow(im[ch])
        spind = spind + 1
    spind = num_fmaps_to_plot + 1
    
    for layer in convlayers:
        # shape is batch_size, num_filters, x,y 
        fmaps = get_output(layer,x ).eval()
        print fmaps.shape
        for fmap_ind in range(num_fmaps_to_plot):
            p1 = plt.subplot(num_layers + 1,num_fmaps_to_plot, spind )
            p1.imshow(fmaps[ind][fmap_ind])
            spind = spind + 1
    
    #pass
    plt.savefig(save_dir +'/fmaps.png')





