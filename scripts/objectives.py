
import matplotlib; matplotlib.use("agg")


from sklearn.metrics import *
import numpy as np



def ams(pred,gt, weights):
    pred =convert_bool_or_conf_to_int(pred)
    br = 10
    #weighted true positives
    s = sum([weights[i] if gt[i]==1. and pred[i]==1. else 0. for i in range(gt.shape[0])])

    #weighted false postivies
    b = sum([weights[i] if gt[i]==0. and pred[i]==1. else 0. for i in range(gt.shape[0])])

    ams = np.sqrt(2 * ((s+b+br)*np.log(1 + (s / (b+br))) - s))[0]
    return dict(ams=ams)

def bg_rej_sig_eff(pred,gt,weights):
    
    pred = convert_bool_or_conf_to_int(pred)
    #of the real backgrounds, how many did you guess were backgrounds
    preds_bg = pred[gt==0.]
    num_bg_rej = preds_bg[preds_bg ==0.].shape[0]
    
    
    #how many actual backgrounds
    num_bg = gt[gt==0].shape[0]
    

    
    #percent of backgrounds guessed as bg (recall for bg)
    bg_rej = float(num_bg_rej) / num_bg
    
    # of the signals, how many did you guess as signal
    preds_sig = pred[gt==1.]
    num_sig_sel = preds_sig[preds_sig==1.].shape[0]
    

    #how many actual signals
    num_sig = gt[gt==1].shape[0]

    sig_eff = float(num_sig_sel) / num_sig
    
    return dict(sig_eff=sig_eff, bg_rej=bg_rej)


def sig_eff_at(bg_rej, pred,gt,weights=None):
    roc = roc_vals(pred,gt,weights)
    des_fpr = 1 - bg_rej
    ind = np.searchsorted(roc["fpr"], des_fpr)
    sig_eff = roc["tpr"][ind]
    return {"sig_eff_at_" + str(bg_rej):sig_eff}

def roc_vals(pred, gt, weights=None):
    
    #pred = convert_bool_or_conf_to_int(pred)
    if weights is None:
        fpr, tpr, thresholds = roc_curve(gt, pred)
    else:
        fpr, tpr, thresholds = roc_curve(gt, pred, sample_weight=weights)
    
    return dict(fpr=fpr, tpr=tpr, thresholds=thresholds)
        

    
    
    
def convert_bool_or_conf_to_int(pred):
    #convert boolean to int/float
    pred = 1*pred
    
    #convert confidences to decisions (1 or 0)
    pred = np.round(pred)
    
    return pred
    



if __name__ == "__main__":
    num = 10000
    test_cut = np.asarray(int((num*0.75))*[True] + int((num*.25))*[False])
    test_gt = np.random.randint(0,2,num)
    test_pred = np.concatenate((np.random.random(num/2) , test_gt[num/2:]))
    test_w = np.random.random(num)
    
    

if __name__ == "__main__":
    ws = [None, test_w]
    preds = [test_pred, test_cut]
    gt = test_gt
    weights = test_w
    weights=None
    pred = test_pred
    print ams(pred,gt, weights)
    print bg_rej_sig_eff(pred,gt,weights)
    
    print sig_eff_at(0.9996, pred,gt,weights)
    d= roc_vals(pred, gt, weights)
            
            

    from matplotlib import pyplot as plt


    plt.plot(d["fpr"], d["tpr"])


