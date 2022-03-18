# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class DeterministicIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices_list):
        ctx.x = x
        ctx.indices_list = indices_list
        return x[indices_list[0], indices_list[1], :, indices_list[2], indices_list[3]]

    @staticmethod
    def backward(ctx, grad_output):
        tmp = torch.zeros_like(ctx.x)
        ind0, ind1, ind2, ind3 = ctx.indices_list
        tmp[ind0, ind1, :, ind2, ind3] = grad_output
        return tmp, None


# @torchsnooper.snoop(output='/data/wyh/yolor/yolorDebug_1P640.txt')
def compute_loss(p, targets, model):  # predictions, targets, model
    device = targets.device

    targets = targets.T
    for i in range(len(p)):
        p[i] = p[i].permute(0, 1, 4, 2, 3) #(6, 3, 80, 80, 85)->(6, 3, 85, 80, 80)

    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)
    tcls, tbox, indices, anchors, targets_mask, targets_sum_mask = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction='sum').to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction='mean').to(device)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    nt = 0  # number of targets
    np = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    balance = [4.0, 1.0, 0.5, 0.4, 0.1] if np == 5 else balance
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        allmask = targets_mask[i]
        sum_mask = targets_sum_mask[i]
        # tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
        tobj = torch.zeros_like(pi[:, :, 0, :, :]).to(device)  # target obj
        
        nb = b.shape[0]  # number of targets
        if sum_mask.item() > 0:
            nt += nb  # cumulative targets
            # ps = pi[b, a,:, gj, gi]  # prediction subset corresponding to targets
            ps = DeterministicIndex.apply(pi, (b, a, gj, gi)).permute(1, 0).contiguous()
            # GIoU
            pxy = ps.index_select(0, torch.tensor([0, 1], device=device))
            pwh = ps.index_select(0, torch.tensor([2, 3], device=device))

            pxy = pxy.sigmoid() * 2. - 0.5
            pwh = (pwh.sigmoid() * 2) ** 2 * (anchors[i].T)
            pbox = torch.cat((pxy, pwh), 0)  # predicted box
            giou = bbox_iou(pbox, tbox[i], x1y1x2y2=False, GIoU=True)
            giou = giou * (allmask) + (1. - allmask)
            lbox += (1.0 - giou).sum() / (sum_mask) # giou loss
            # Obj
            giou = giou * (allmask)
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio
            
            # Class
            if model.nc > 1:  # cls loss (only if multiple classes)
                tmp = ps[5:, :]
                tmp = tmp * (allmask) - (1.- allmask) * 50.
                t = torch.full_like(tmp, cn).to(device)  # targets
                range_nb = torch.arange(nb, device=device).long()
                t[tcls[i], range_nb] = cp

                t = t * (allmask)
                lcls += (BCEcls(tmp, t) / (sum_mask * t.shape[0]).float()) # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        # lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss
        lobj += BCEobj(pi[:, :, 4, :, :], tobj) * balance[i]  # obj loss

    s = 3 / np  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if np >= 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    na, nt, device, batch_size = 3, targets.shape[1], targets.device, p[0].shape[0]
    
    # align targets in batch size
    nt_max = 32 * batch_size
    while nt > nt_max:
        nt_max *= 2
        print('**************** nt max=', nt_max)
    max_target = torch.zeros(6, nt_max, device=device)  # (6, nt)
    for i in range(6):
        try:
            max_target[i, :nt] = targets[i, :]
            # print('Check------', max_target.shape, max_target.device, device)
            # print('Check------', targets.shape, targets.device, device)
        except Exception as e: 
            print(e)
            # print('Check------', max_target.shape, max_target.device, device)
            # print('Check------', targets.shape, targets.device, device)

    tcls, tbox, indices, anch, targets_mask, targets_sum_mask = [], [], [], [], [], []
    gain = torch.ones(6, device=device)  # normalized to gridspace gain
    off_list = [
        torch.tensor([[1.], [0.]], device=device),
        torch.tensor([[0.], [1.]], device=device),
        torch.tensor([[-1.], [0.]], device=device),
        torch.tensor([[0.], [-1.]], device=device)
    ]
    # # create indices with anchor and max_target
    # # anchor tensor, same as .repeat_interleave(nt)  (x, 3)
    at = torch.arange(na).view(na, 1).repeat(1, nt_max)
    a = at.view(-1)
    a = torch.cat((a, a, a, a, a), 0)
    
    g = 0.5 # offset
    # multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    multi_gpu = is_parallel(model)
    for i, j in enumerate(model.module.yolo_layers if multi_gpu else model.yolo_layers):
        # get number of grid points and anchor vec for this yolo layer
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        # iou of targets-anchors b,a,c,y,x-> b,a,y,x,c
        # gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]].float()  # xyxy gain
        gain[2:] = torch.tensor(p[i].shape)[[4, 3, 4, 3]].float()  # xyxy gain
        
        # Match targets to anchors
        t, offsets = max_target * gain[:, None], 0
        allmask = torch.zeros((na * nt_max)).to(device)
        sum_mask = torch.zeros((1)).to(device)
        if nt:
            r = t[None, 4:6, :] / anchors[..., None]  # wh ratio
            fmask = torch.max(r, 1. / r).max(1)[0] < model.hyp['anchor_t']  # compare
            fmask = fmask.view(1, -1)
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            t = t.repeat(1, 1, na).view(6, -1)  # filter

            # overlaps
            gxy = t.index_select(0, torch.tensor([2, 3], device=device)) 
            z = torch.zeros_like(gxy)

            jk = (gxy % 1. < g) & (gxy > 1.)
            lm = (gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]][:, None] - 1.))
            jk, lm = jk&fmask, lm&fmask
            allmask = torch.cat((fmask, jk, lm), 0).view(1, -1).float()
            t = torch.cat((t, t, t, t, t), 1)
            offsets = torch.cat((z, z + off_list[0], z + off_list[1], z + off_list[2], z + off_list[3]), 1) * g

            # print('----------------------------------------------------------------------------------')
            # print('a.shape, t.shape:')
            # print(a.shape, t.shape)
            # print('gxy.shape, offsets.shape')
            # print(gxy.shape, offsets.shape)
            # print('fmask.shape, allmask.shape, jk, lm:')
            # print(fmask.shape, allmask.shape, jk.shape, lm.shape)
            # print('----------------------------------------------------------------------------------')

            sum_mask = allmask.sum()
            t = t * allmask

        # Define
        b = t.index_select(0, torch.tensor([0], device=device)).long().view(-1)   #(3072 * 5)
        c = t.index_select(0, torch.tensor([1], device=device)).long().view(-1)   #(3072 * 5)
        gxy = t.index_select(0, torch.tensor([2, 3], device=device)) #(2, 3072 * 5)
        gwh = t.index_select(0, torch.tensor([4, 5], device=device)) #(2, 3072 * 5)
        gij = gxy - offsets
        gij2 = gij.long()
        gi = gij2.index_select(0, torch.tensor([0], device=device)).view(-1) #(2, 3072 * 5)
        gj = gij2.index_select(0, torch.tensor([1], device=device)).view(-1) #(2, 3072 * 5)

        # Append
        # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij2.float(), gwh), 0))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
        targets_mask.append(allmask)
        targets_sum_mask.append(sum_mask)

    return tcls, tbox, indices, anch, targets_mask, targets_sum_mask

# def build_targets(p, targets, model):
#     nt = targets.shape[0]  # number of anchors, targets
#     tcls, tbox, indices, anch, targets_mask, targets_sum_mask = [], [], [], [], [], []
#     gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
#     off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

#     # align targets in batch size
#     batch_size = p[0].shape[0]
#     nt_max = 32 * batch_size
#     while nt > nt_max:
#         nt_max *= 2
#         print('**************** nt max=', nt_max)
#     max_target = torch.zeros(nt_max, 6, device=targets.device)  # (nt,6)
#     for i in range(6):
#         max_target[:nt, i] = targets[:, i]

#     g = 0.5  # offset
#     multi_gpu = is_parallel(model)
#     for i, jj in enumerate(model.module.yolo_layers if multi_gpu else model.yolo_layers):
#         # get number of grid points and anchor vec for this yolo layer
#         anchors = model.module.module_list[jj].anchor_vec if multi_gpu else model.module_list[jj].anchor_vec
#         gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

#         # Match targets to anchors
#         a, t, offsets = [], max_target * gain, 0
        
#         if nt:
#             na = anchors.shape[0]  # number of anchors
#             allmask = torch.zeros((na * nt_max)).to(targets.device)
#             sum_mask = torch.zeros((1)).to(targets.device)
#             at = torch.arange(na).view(na, 1).repeat(1, nt_max)  # anchor tensor, same as .repeat_interleave(nt)
#             r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
#             fmask = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
#             # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
#             a, t = at[fmask], t.repeat(na, 1, 1)[fmask]  # filter

#             print('----------------------------------------------------------------------------------')
#             print('a.shape, at.shape, t.shape:')
#             print(a.shape, at.shape, t.shape)
#             print('----------------------------------------------------------------------------------')

#             # overlaps
#             gxy = t[:, 2:4]  # grid xy
#             z = torch.zeros_like(gxy)
#             j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#             l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
            
#             print(a.shape, a[j].shape, a[k].shape, a[l].shape, a[m].shape)
#             print(t.shape, t[j].shape, t[k].shape, t[l].shape, t[m].shape)
            
#             a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
#             offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g
            
#             allmask = torch.cat((j, k, l, m), 1).float()
#             sum_mask = allmask.sum()
            
#             print('----------------------------------------------------------------------------------')
#             print('a.shape, t.shape:')
#             print(a.shape, t.shape)
#             print('gxy.shape, offsets.shape')
#             print(gxy.shape, offsets.shape)
#             print('fmask.shape, allmask.shape, j, k, l, m:')
#             print(fmask.shape, allmask.shape, j.shape, k.shape, l.shape, m.shape)
#             print('----------------------------------------------------------------------------------')
            
#             t = t * allmask

#         # Define
#         b, c = t[:, :2].long().T  # image, class
#         gxy = t[:, 2:4]  # grid xy
#         gwh = t[:, 4:6]  # grid wh
#         gij = (gxy - offsets).long()
#         gi, gj = gij.T  # grid xy indices

#         # Append
#         #indices.append((b, a, gj, gi))  # image, anchor, grid indices
#         indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
#         tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#         anch.append(anchors[a])  # anchors
#         tcls.append(c)  # class
#         targets_mask.append(allmask)
#         targets_sum_mask.append(sum_mask)

#     return tcls, tbox, indices, anch, targets_mask, targets_sum_mask
