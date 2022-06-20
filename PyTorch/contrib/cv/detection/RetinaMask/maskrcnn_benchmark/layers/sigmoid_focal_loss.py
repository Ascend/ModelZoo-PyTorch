import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

def py_sigmoid_focal_loss(pred,
                          target,
                          gamma=2.0,
                          alpha=0.25):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight

    return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, num_classes, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        num_classes = logits.size(1) + 1
        valid = (targets != -1)[:, None]
        targets = F.one_hot(targets.abs(), num_classes=num_classes)
        targets = targets[:, 1:num_classes]
        loss = py_sigmoid_focal_loss(
            logits, targets, self.gamma, self.alpha
        )
        loss = loss * valid

        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "num_classes=" + str(self.num_classes)
        tmpstr += ", gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
