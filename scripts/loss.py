import torch
import logging
from torch import nn
from torch.nn import functional as F
class AMSoftmax(nn.Module):

    '''
    Additve Margin Softmax as proposed in:
    https://arxiv.org/pdf/1801.05599.pdf
    Implementation Extracted From
    https://github.com/clovaai/voxceleb_trainer/blob/master/loss/cosface.py
    '''

    def __init__(self, in_feats, n_classes, m=0.3, s=15, annealing=False):
        super(AMSoftmax, self).__init__()
        self.in_feats = in_feats
        self.m = m
        self.s = s
        self.annealing = annealing
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)
        self.annealing=annealing
        

    def getAnnealedFactor(self,step):
        alpha = self.__getAlpha(step) if self.annealing else 0.
        return 1/(1+alpha)

    def __getAlpha(self,step):
        return max(0, 1000./(pow(1.+0.0001*float(step),2.)))        

    def __getCombinedCosth(self, costh, costh_m, step):

        alpha = self.__getAlpha(step) if self.annealing else 0.
        costh_combined = costh_m + alpha*costh
        return costh_combined/(1+alpha)

    def forward(self, x, label=None, step=0):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_combined = self.__getCombinedCosth(costh, costh_m, step)
        costh_m_s = self.s * costh_combined

        return costh, costh_m_s 
 
class FocalSoftmax(nn.Module):
    ''' 
    Focal softmax as proposed in:
    "Focal Loss for Dense Object Detection"
    by T-Y. Lin et al.
    https://github.com/foamliu/InsightFace-v2/blob/master/focal_loss.py
    '''
    def __init__(self, gamma=2):
        super(FocalSoftmax, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    
class OrdinalRegression(nn.Module):
    '''
    Ordinal Regressor for Age recog. Check: https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99
    '''

    def __init__(self):
        super(OrdinalRegression, self).__init__()
        self.mse = nn.MSELoss(reduction='none')


    def forward(self, predictions: [[float]], targets: [float]):
        # Create out modified target with [batch_size, num_labels] shape
        modified_target = torch.zeros_like(predictions)

        # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
        for i, target in enumerate(targets):
            modified_target[i, 0:target+1] = 1
        
        return self.mse(predictions, modified_target).sum(axis=1).mean()

class OrdinalRegressionWeighted(nn.Module):

    def __init__(self, b_w):
        super(OrdinalRegressionWeighted, self).__init__()
        self.balanced_weights = b_w
        #self.w_m_l = weighted_mse_loss()
    def forward(self, predictions:  [[float]], targets: [float]):
        # Create out modified target with [batch_size, num_labels] shape
        modified_target = torch.zeros_like(predictions)
        # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
        for i, target in enumerate(targets):
            modified_target[i, 0:target+1] = 1

        return weighted_mse_loss(predictions,modified_target,self.balanced_weights)

    def weighted_mse_loss(input, target, weight):
        weight_v = torch.zeros_like(target)
        for i, v in enumerate(target):
            weight_v[i] = 1 / weight[v.long()]
        return (weight_v * (input - target) ** 2).mean() / weight_v.mean()

def weighted_mse_loss(input,target,weight):
    weight_v = torch.zeros_like(target)
    for i,v in enumerate(target):
        weight_v[i] = 1 / weight[v.long()]
    return (weight_v * (input - target) ** 2).mean() / weight_v.mean()

class OrdinalRegressionTuned(nn.Module):
    def __init__(self):
        super(OrdinalRegressionTuned, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.alfa = torch.ones(1, requires_grad=True).to(dtype=torch.long).to(device='cuda')
        #self.lossl1 = nn.L1Loss(reduction='none')


    def forward(self, predictions: [[float]], targets: [float]):
        # Create out modified target with [batch_size, num_labels] shape

        
        modified_predictions = torch.zeros_like(predictions)
        
        for i, target in enumerate(targets):
            for j in range(0,6):
                #modified_predictions[i,j] = predictions[i,j]*abs(target-j+1)*abs(target-j+1)
                modified_predictions[i,j] = predictions[i,j]*abs(target-j+1)**self.alfa
                
        modified_predictions = modified_predictions.to(dtype=torch.float32)
        targets = F.one_hot(targets, num_classes=6).to(dtype=torch.float32)
        return self.mse(modified_predictions, targets).sum(axis=1).mean().to(dtype=torch.float32)
        #return self.lossl1(modified_predictions, targets).sum(axis=1).mean().to(dtype=torch.float32)
        

class OrdinalModifiedRegression(nn.Module):
    '''
    Ordinal Regressor for Age recog. Check: https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99
    '''

    def __init__(self):
        super(OrdinalModifiedRegression, self).__init__()
        self.mse = nn.MSELoss(reduction='none')


    def forward(self, predictions: [[float]], targets: [float]):
        # Create out modified target with [batch_size, num_labels] shape

        preds = torch.max(predictions, 1)[1]
        j=0

        for pred, label in zip(preds,targets):
            if pred == label-1 or pred == label+1:
                preds[j]=label
            j=j+1

        modified_target = torch.zeros_like(predictions)
        modified_predictions = torch.zeros_like(predictions)
        # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
        for i, target in enumerate(targets):
            modified_target[i, 0:target+1] = 1
        
        for i, prediction in enumerate(preds):
            modified_predictions[i,0:prediction+1] = 1
        
        return self.mse(modified_predictions, modified_target).sum(axis=1).mean()