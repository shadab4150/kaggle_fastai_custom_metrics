import sklearn 
from fastai.vision import *
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import torch

class column_mean_aucroc(Callback):
    '''
    Column wise mean AUCROC:
    It's for one hot encoded targets only.
    The AUCROC is computed for each target column, 
    and the mean of these values is calculated for the score.
    '''
    def __init__(self):
        self.average = 'micro'

    def on_epoch_begin(self, **kwargs):
        self.preds = None
        self.target = None
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        if self.preds is None or self.target is None:
            self.preds = last_output.cpu()
            self.target = last_target.cpu()
        else:
            self.preds = np.append(self.preds, last_output.cpu(), axis=0)
            self.target = np.append(self.target, last_target.cpu(), axis=0)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        col = self.preds.shape[1]
        res = np.mean([sklearn.metrics.roc_auc_score(self.target[:,i], self.preds[:,i],average=self.average) for i in range(col)])
        return add_metrics(last_metrics, res)


class AvgSpearman(Callback):
    '''
     Mean column-wise Spearman's correlation coefficient. 
     The Spearman's rank correlation is computed for each target column, 
     and the mean of these values is calculated for the score.
    '''
    def on_epoch_begin(self, **kwargs):
        self.preds = None
        self.target = None
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        if self.preds is None or self.target is None:
            self.preds = last_output.cpu()
            self.target = last_target.cpu()
        else:
            self.preds = np.append(self.preds, last_output.cpu(), axis=0)
            self.target = np.append(self.target, last_target.cpu(), axis=0)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        spearsum = 0
        for col in range(self.preds.shape[1]):
            spearsum += spearmanr(self.preds[:,col], self.target[:,col]).correlation
        res = spearsum / (self.preds.shape[1] + 1)
        return add_metrics(last_metrics, res)



def mask_accuracy(input, target):
    '''
    Mask accuracy for segmentation task
    '''
    void_code = -1
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()



class GAP_vector(Callback):
       
    ''''
    https://www.kaggle.com/c/landmark-recognition-2019/discussion/90752#latest-525286
    https://docs.fast.ai/callback.html

    "Wrap a `func` in a callback for metrics computation."
   
Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition. 
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth

    Returns:
        GAP score
        
    Fastai Provides:
        last_output: contains the last output spitted by the model (eventually updated by a callback)
        last_loss: contains the last loss computed (eventually updated by a callback)
        last_target: contains the last target that got through the model (eventually updated by a callback)
    '''
    _order = -20
    
    def __init__(self):
        nlearn:Learner
        name:str='GAP_vector'
            
    def on_epoch_begin(self, **kwargs):
        # Creates empty list for predictions and targets
        self.targs, self.preds,  self.loss = LongTensor([]), Tensor([]),Tensor([])
        
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        # gets the predictions and targets for each batch

        # Indicies are assoicated with class prediction
        _, indices = torch.max(last_output, 1)
        indices = torch.as_tensor(indices, dtype=torch.float)#, device=device)
        # Finds the class with highest probability
        last_output = F.softmax(last_output, dim=1)[:,-1]
        
        # Appends the list with the predicted
        self.preds = torch.cat((self.preds, indices.cpu()))
        # Appends the list with the target
        self.targs = torch.cat((self.targs, last_target.cpu().long()))
        # Appends the list with the probability
        
        self.loss = torch.cat((self.loss, last_output.cpu()))

        
    def on_epoch_end(self, last_output, last_loss, last_metrics, **kwargs):
        "Set the final result for GAP Score`."
        # Creates the dataframe
        x = pd.DataFrame({'pred': self.preds, 'conf': self.loss, 'true': self.targs})
        # sorts the values by confidence
        x.sort_values('conf', ascending=False, inplace=True, na_position='last')
        # Makes a column for the number correct. Is true if the prediction is the same as target
        x['correct'] = (x.true == x.pred).astype(int)
        # creates column for predictions
        x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
        # gets the total score
        x['term'] = x.prec_k * x.correct
        # divides by the count of true
        gap = x.term.sum() / x.true.count()
        return add_metrics(last_metrics, gap)


class column_mean_logloss(Callback):
    '''
    Column wise mean AUCROC:
    It's for one hot encoded targets only.
    The AUCROC is computed for each target column, 
    and the mean of these values is calculated for the score.
    '''

    def on_epoch_begin(self, **kwargs):
        self.preds = None
        self.target = None
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        if self.preds is None or self.target is None:
            self.preds = last_output.cpu()
            self.target = last_target.cpu()
        else:
            self.preds = np.append(self.preds, last_output.cpu(), axis=0)
            self.target = np.append(self.target, last_target.cpu(), axis=0)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        col = self.preds.shape[1]
        res = np.mean([sklearn.metrics.log_loss(self.target[:,i], self.preds[:,i]) for i in range(col)])
        return add_metrics(last_metrics, res)



def weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights =        [       2,   1]
    
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_valid, pos_label=1)
    
    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    
    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)
    
    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min # normalize such that curve starts at y=0
        score = sklearn.metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric
        
    return competition_metric / normalization



class alaska_weighted_auc(Callback):
    '''
    Weighted AUCROC:
    For Alaska Steganalysis Competition
    Competiton : https://www.kaggle.com/c/alaska2-image-steganalysis/overview/evaluation
    '''

    def on_epoch_begin(self, **kwargs):
        self.preds = None
        self.target = None
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        if self.preds is None or self.target is None:
            self.preds = last_output.cpu()
            self.target = last_target.cpu()
        else:
            self.preds = np.append(self.preds, last_output.cpu(), axis=0)
            self.target = np.append(self.target, last_target.cpu(), axis=0)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        res = weighted_auc(self.target, self.preds[:,1])
        return add_metrics(last_metrics, res)




class weighted_mae(Callback):
    '''
    Feature-Weighted, Normalized Absolute Errors
    For Competition : https://www.kaggle.com/c/trends-assessment-prediction/overview/evaluation
    '''
    def __init__(self):
        self.wgt = [0.3, 0.175, 0.175, 0.175, 0.175]

    def on_epoch_begin(self, **kwargs):
        self.preds = None
        self.target = None
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        if self.preds is None or self.target is None:
            self.preds = last_output.cpu()
            self.target = last_target.cpu()
        else:
            self.preds = np.append(self.preds, last_output.cpu(), axis=0)
            self.target = np.append(self.target, last_target.cpu(), axis=0)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        res = sum([sklearn.metrics.mean_absolute_error(self.target[:,i], self.preds[:,i])*self.wgt[i] for i in range(5)])
        return add_metrics(last_metrics, res)




def print_all_metrics():
    from prettytable import PrettyTable
    metrics = [['column_mean_logloss','column_mean_aucroc','weighted_mae','alaska_weighted_auc','AvgSpearman','GAP_vector'],
               ['https://www.kaggle.com/c/plant-pathology-2020-fgvc7',
                'https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge',
                'https://www.kaggle.com/c/trends-assessment-prediction',
                'https://www.kaggle.com/c/alaska2-image-steganalysis','https://www.kaggle.com/c/google-quest-challenge',
                'https://www.kaggle.com/c/landmark-recognition-2020']]
    
    t = PrettyTable(['Competition URL', 'Metric'])
    for i in range(len(metrics[0])):
        t.add_row([metrics[1][i],metrics[0][i]])
    print(t)
    return 

def get_metrics(metric):
	metrics = {'column_mean_aucroc' : column_mean_aucroc(),
				'column_mean_logloss' : column_mean_logloss(),
				'weighted_mae' : weighted_mae(),
				'alaska_weighted_auc': alaska_weighted_auc(),
				'mask_accuracy' : mask_accuracy,
				'GAP_vector' : GAP_vector(),
				'AvgSpearman' : AvgSpearman()}
	return metrics[metric]
