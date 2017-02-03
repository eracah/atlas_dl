
import matplotlib; matplotlib.use("agg")


from lasagne.layers import get_all_layers, count_params
import numpy as np



def print_network(networks, kwargs):
    def _print_network(network):
        kwargs['logger'].info("\n")
        for layer in get_all_layers(network):
            kwargs['logger'].info(str(layer) +' : ' + str(layer.output_shape))
        kwargs['logger'].info("Total Parameters: " + str(count_params(layer)))
        kwargs['logger'].info("\n")
    for net in networks.values():
        _print_network(net)



def print_results(kwargs, epoch, metrics):
    if not kwargs["test"]:
        kwargs['logger'].info("Epoch {} of {} took {:.3f}s".format(epoch + 1, kwargs['num_epochs'],
                                                                  metrics["train_time"][-1]))
        for typ in ["train", "validation"]:
            if typ == "validation":
                kwargs['logger'].info("\tValidation took {:.3f}s".format(metrics["validation_time"][-1]))
            for k,v in metrics.iteritems():
                #print k,v
                val = v[-1][0] if isinstance(v[-1], list) or isinstance(v[-1], np.ndarray)  else v[-1]
                if typ in k[:4] and "time" not in k:
                    if "ams" not in k and "loss" not in k:
                        kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f} %".format(val * 100))

                    else:
                        kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f}".format(val))
    else:
            for k,v in metrics.iteritems():
                print(k,v)
        
        

