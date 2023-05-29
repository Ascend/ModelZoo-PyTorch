# -*- coding: UTF-8 -*-

'''
Label Smoothing described in "Rethinking the Inception Architecture for Computer Vision"
Ref: https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/nn/loss.py
     https://github.com/whr94621/NJUNMT-pytorch/blob/master/src/modules/criterions.py
'''

import torch
from torch import nn
from torch.autograd import Variable

class LabelSmoothingLoss(nn.Module):
    '''
    Label Smoothing Loss function
    '''

    def __init__(self, classes_num, label_smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.classes_num = classes_num
        self.dim = dim
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        smooth_label = torch.empty(size=pred.size(), device=target.device)
        smooth_label.fill_(self.label_smoothing / (self.classes_num - 1))
        smooth_label.scatter_(1, target.data.unsqueeze(1), self.confidence)
        #return torch.mean(torch.sum(-smooth_label * pred, dim=self.dim))
        return self.criterion(pred, Variable(smooth_label, requires_grad=False))

if __name__ == "__main__":

    loss1 = LabelSmoothingLoss(5, 0.0)
    
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0], [0, 0.9, 0.2, 0.1, 0], [1, 0.2, 0.7, 0.1, 0]])
    v1 = loss1(Variable(predict), Variable(torch.LongTensor([2, 1, 0])))
    print(v1)
    
    loss2 = nn.CrossEntropyLoss()
    v2 = loss2(Variable(predict), Variable(torch.LongTensor([2, 1, 0])))
    print(v2.data)