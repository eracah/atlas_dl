
import matplotlib; matplotlib.use("agg")


import sys
from scripts.metrics.objectives import roc_vals, bg_rej_sig_eff
from matplotlib import pyplot as plt
from os.path import join, exists
from scripts.util import makedir_if_not_there



def plot_roc_curve(pred, y,w, cuts, type_, save_path):
    def _plot_roc_curve(name="",xlim=[0,1], ylim=[0,1] ):

            #signal preds
            roc = roc_vals(pred, y, w)
            roc_path = join(save_path, "roc_curves")
            makedir_if_not_there(roc_path)
            plt.clf()
            plt.figure(1)
            plt.clf()
            plt.title('%s ROC Curve %s' %(type_, name))
            plt.plot(roc["fpr"], roc["tpr"])


            cuts_results = bg_rej_sig_eff(cuts,y,w)
            cuts_bgrej, cuts_sigeff = cuts_results["bg_rej"], cuts_results["sig_eff"]
            plt.scatter(1-cuts_bgrej, cuts_sigeff)
            plt.ylim(ylim)
            plt.xlim(xlim)
            plt.legend( loc = 'center left', bbox_to_anchor = (1.0, 0.5),
               ncol=2)
            plt.xlabel("False Positive Rate (1- BG rejection)")
            plt.ylabel("True Positive Rate (Signal Efficiency)")
            plt.savefig("%s/%s_roc_curve_%s.png"%(roc_path,type_,name))
            #pass
            plt.clf()
            
    _plot_roc_curve()
    for i in range(1,6):
        _plot_roc_curve(name="zoomed"+ str(i), xlim=[0,10**-i])

        
        

def plot_learn_curve(metrics, save_path):
    def _plot_learn_curve(type_):
        plt.clf()
        plt.figure(1)
        plt.clf()
        plt.title('Train/Val %s' %(type_))
        plt.plot(metrics['tr_' + type_], label='train ' + type_)
        plt.plot(metrics['val_' + type_], label='val ' + type_)
        plt.legend( loc = 'center left', bbox_to_anchor = (1.0, 0.5),
           ncol=2)

        curves_path = join(save_path, "learn_curves")
        makedir_if_not_there(curves_path)
        plt.savefig("%s/%s_learning_curve.png"%(curves_path,type_))
        if "loss" in type_:
            pass
        plt.clf()
        
        
    for k in metrics.keys():
        if "time" not in k and "phys" not in k:
            type_ = '_'.join(k.split("_")[1:])
            _plot_learn_curve(type_)


