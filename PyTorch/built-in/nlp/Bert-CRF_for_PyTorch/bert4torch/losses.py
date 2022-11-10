# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ast import arg
from tracemalloc import start
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean',ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                           ignore_index=self.ignore_index)


class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, y_true):
        """ y_true ([Tensor]): [..., num_classes]
            y_pred ([Tensor]): [..., num_classes]
        """
        y_pred = (1-2*y_true) * y_pred
        y_pred_pos = y_pred - (1-y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12

        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()


class SparseMultilabelCategoricalCrossentropy(nn.Module):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    def __init__(self, mask_zero=False, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.mask_zero = mask_zero
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)
        if self.mask_zero:
            infs = zeros + float('inf')
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
        if self.mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)  # a
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss  # b-a
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), self.epsilon, 1)  # 1-exp(b-a)
        neg_loss = all_loss + torch.log(aux_loss)  # a + log[1-exp(b-a)]
        return pos_loss + neg_loss


class ContrastiveLoss(nn.Module):
    """对比损失：减小正例之间的距离，增大正例和反例之间的距离
    公式：labels * distance_matrix.pow(2) + (1-labels)*F.relu(margin-distance_matrix).pow(2)
    https://www.sbert.net/docs/package_reference/losses.html
    """
    def __init__(self, margin=0.5, size_average=True, online=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.online = online

    def forward(self, distances, labels, pos_id=1, neg_id=0):
        if not self.online:
            losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
            return losses.mean() if self.size_average else losses.sum()
        else:
            negs = distances[labels == neg_id]
            poss = distances[labels == pos_id]

            # select hard positive and hard negative pairs
            negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]
            
            positive_loss = positive_pairs.pow(2).sum()
            negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
            return positive_loss + negative_loss


class RDropLoss(nn.Module):
    '''R-Drop的Loss实现，官方项目：https://github.com/dropreg/R-Drop
    '''
    def __init__(self, alpha=4, rank='adjacent'):
        super().__init__()
        self.alpha = alpha
        # 支持两种方式，一种是奇偶相邻排列，一种是上下排列
        assert rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rank = rank
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_rdrop = nn.KLDivLoss(reduction='none')

    def forward(self, *args):
        '''支持两种方式: 一种是y_pred, y_true, 另一种是y_pred1, y_pred2, y_true
        '''
        assert len(args) in {2, 3}, 'RDropLoss only support 2 or 3 input args'
        # y_pred是1个Tensor
        if len(args) == 2:
            y_pred, y_true = args
            loss_sup = self.loss_sup(y_pred, y_true)  # 两个都算

            if self.rank == 'adjacent':
                y_pred1 = y_pred[1::2]
                y_pred2 = y_pred[::2]
            elif self.rank == 'updown':
                half_btz = y_true.shape[0] // 2
                y_pred1 = y_pred[:half_btz]
                y_pred2 = y_pred[half_btz:]
        # y_pred是两个tensor
        else:
            y_pred1, y_pred2, y_true = args
            loss_sup = self.loss_sup(y_pred1, y_true)

        loss_rdrop1 = self.loss_rdrop(F.log_softmax(y_pred1, dim=-1), F.softmax(y_pred2, dim=-1))
        loss_rdrop2 = self.loss_rdrop(F.log_softmax(y_pred2, dim=-1), F.softmax(y_pred1, dim=-1))
        return loss_sup + torch.mean(loss_rdrop1 + loss_rdrop2) / 4 * self.alpha


class UDALoss(nn.Module):
    '''UDALoss，使用时候需要继承一下，因为forward需要使用到global_step和total_steps
    https://arxiv.org/abs/1904.12848
    '''
    def __init__(self, tsa_schedule=None, total_steps=None, start_p=0, end_p=1, return_all_loss=True):
        super().__init__()
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_unsup = nn.KLDivLoss(reduction='batchmean')
        self.tsa_schedule = tsa_schedule
        self.start = start_p
        self.end = end_p
        if self.tsa_schedule:
            assert self.tsa_schedule in {'linear_schedule', 'exp_schedule', 'log_schedule'}, 'tsa_schedule config illegal'
        self.return_all_loss = return_all_loss

    def forward(self, y_pred, y_true_sup, global_step, total_steps):
        sup_size = y_true_sup.size(0)
        unsup_size = (y_pred.size(0) - sup_size) // 2

        # 有监督部分, 用交叉熵损失
        y_pred_sup = y_pred[:sup_size]
        if self.tsa_schedule is None:
            loss_sup = self.loss_sup(y_pred_sup, y_true_sup)
        else:  # 使用tsa来去掉预测概率较高的有监督样本
            threshold = self.get_tsa_threshold(self.tsa_schedule, global_step, total_steps, self.start, self.end)
            true_prob = torch.gather(F.softmax(y_pred_sup, dim=-1), dim=1, index=y_true_sup[:, None])
            sel_rows = true_prob.lt(threshold).sum(dim=-1).gt(0)  # 仅保留小于阈值的样本
            loss_sup = self.loss_sup(y_pred_sup[sel_rows], y_true_sup[sel_rows]) if sel_rows.sum() > 0 else 0

        # 无监督部分，这里用KL散度，也可以用交叉熵
        y_true_unsup = y_pred[sup_size:sup_size+unsup_size]
        y_true_unsup = F.softmax(y_true_unsup.detach(), dim=-1)
        y_pred_unsup = F.log_softmax(y_pred[sup_size+unsup_size:], dim=-1)
        loss_unsup = self.loss_unsup(y_pred_unsup, y_true_unsup)
        if self.return_all_loss:
            return loss_sup + loss_unsup, loss_sup, loss_unsup
        else:
            return loss_sup + loss_unsup

    @ staticmethod
    def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
        training_progress = global_step / num_train_steps
        if schedule == "linear_schedule":
            threshold = training_progress
        elif schedule == "exp_schedule":
            scale = 5
            threshold = math.exp((training_progress - 1) * scale)
        elif schedule == "log_schedule":
            scale = 5
            threshold = 1 - math.exp((-training_progress) * scale)
        return threshold * (end - start) + start


class TemporalEnsemblingLoss(nn.Module):
    '''TemporalEnsembling的实现，思路是在监督loss的基础上，增加一个mse的一致性损失loss
       官方项目：https://github.com/s-laine/tempens
       pytorch第三方实现：https://github.com/ferretj/temporal-ensembling
       使用的时候，train_dataloader的shffle必须未False
    '''
    def __init__(self, epochs, max_val=10.0, ramp_up_mult=-5.0, alpha=0.5, max_batch_num=100, hist_device='cpu'):
        super().__init__()
        self.loss_sup = nn.CrossEntropyLoss()
        self.max_epochs = epochs
        self.max_val = max_val
        self.ramp_up_mult = ramp_up_mult
        self.alpha = alpha
        self.max_batch_num = max_batch_num  # 设置未None表示记录全部数据历史，数据量大时耗资源
        self.hist_unsup = []  # 历史无监督logit
        self.hist_sup = []  # 历史监督信息logit
        self.hist_device = hist_device
        self.hist_input_y = []  # 历史监督标签y
        assert (self.alpha >= 0) & (self.alpha < 1)  # 等于1的时候upata写分母为0

    def forward(self, y_pred_sup, y_pred_unsup, y_true_sup, epoch, bti):
        self.same_batch_check(y_pred_sup, y_pred_unsup, y_true_sup, bti)
        
        if (self.max_batch_num is None) or (bti < self.max_batch_num):
            self.init_hist(bti, y_pred_sup, y_pred_unsup)  # 初始化历史
            sup_ratio = float(len(y_pred_sup)) / (len(y_pred_sup) + len(y_pred_unsup))  # 监督样本的比例
            w = self.weight_schedule(epoch, sup_ratio)
            sup_loss, unsup_loss = self.temporal_loss(y_pred_sup, y_pred_unsup, y_true_sup, bti)

            # 更新
            self.hist_unsup[bti] = self.update(self.hist_unsup[bti], y_pred_unsup.detach(), epoch)
            self.hist_sup[bti] = self.update(self.hist_sup[bti], y_pred_sup.detach(), epoch)
            # if bti == 0:  $ 用于检查每个epoch数据顺序是否一致
            #     print(w, sup_loss.item(), w * unsup_loss.item())
            #     print(y_true_sup)
            return sup_loss + w * unsup_loss, sup_loss, w * unsup_loss
        else:
            return self.loss_sup(y_pred_sup, y_true_sup)

    def same_batch_check(self, y_pred_sup, y_pred_unsup, y_true_sup, bti):
        '''检测数据的前几个batch必须是一致的, 这里写死是10个
        '''
        if bti >= 10:
            return
        if bti >= len(self.hist_input_y):
            self.hist_input_y.append(y_true_sup.to(self.hist_device))
        else:  # 检测
            err_msg = 'TemporalEnsemblingLoss requests the same sort dataloader, you may need to set train_dataloader shuffle=False'
            assert self.hist_input_y[bti].equal(y_true_sup.to(self.hist_device)), err_msg
        
    def update(self, hist, y_pred, epoch):
        '''更新历史logit，利用alpha门控来控制比例
        '''
        Z = self.alpha * hist.to(y_pred) + (1. -self.alpha) * y_pred
        output = Z * (1. / (1. - self.alpha ** (epoch + 1)))
        return output.to(self.hist_device)

    def weight_schedule(self, epoch, sup_ratio):
        max_val = self.max_val * sup_ratio
        if epoch == 0:
            return 0.
        elif epoch >= self.max_epochs:
            return max_val
        return max_val * np.exp(self.ramp_up_mult * (1. - float(epoch) / self.max_epochs) ** 2)

    def temporal_loss(self, y_pred_sup, y_pred_unsup, y_true_sup, bti):
        # MSE between current and temporal outputs
        def mse_loss(out1, out2):
            quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
            return quad_diff / out1.data.nelement()
        
        sup_loss = self.loss_sup(y_pred_sup, y_true_sup)
        # 原来实现是sup和unsup作为一个tensor，整体计算的，这里由于是拆分成两个tensor，因此分开算
        unsup_loss = mse_loss(y_pred_unsup, self.hist_unsup[bti].to(y_pred_unsup))
        unsup_loss += mse_loss(y_pred_sup, self.hist_sup[bti].to(y_pred_sup))
        return sup_loss, unsup_loss
    
    def init_hist(self, bti, y_pred_sup, y_pred_unsup):
        if bti >= len(self.hist_sup):
            self.hist_sup.append(torch.zeros_like(y_pred_sup).to(self.hist_device))
            self.hist_unsup.append(torch.zeros_like(y_pred_unsup).to(self.hist_device))