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

import torch
if torch.__version__ >= '1.8':
    import torch_npu

import torch.nn.functional as F
# from SSD import _C as C
import numpy as np
import itertools

from math import sqrt

def calc_iou_tensor(box1, box2):
    """ 
        Calculation of IoU based on two boxes tensor,
        Reference to https://github.com/kuangliu/pytorch-ssd
        input:
            box1 (N, 4)
            box2 (M, 4)
        output:
            IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(be1[:, :, :2], be2[:, :, :2])
    rb = torch.min(be1[:, :, 2:], be2[:, :, 2:])

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:, :, 0] * delta[:, :, 1]

    delta1 = be1[:, :, 2:] - be1[:, :, :2]
    area1 = delta1[:, :, 0] * delta1[:, :, 1]
    delta2 = be2[:, :, 2:] - be2[:, :, :2]
    area2 = delta2[:, :, 0] * delta2[:, :, 1]

    iou = intersect / (area1 + area2 - intersect)
    return iou

class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, \
                       scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps = steps
        self.scales = scales

        fk = fig_size/np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float).cpu()
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5*self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5*self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5*self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5*self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes

# This class is from https://github.com/kuangliu/pytorch-ssd
class Encoder(object):
    """
        Inspired by https://github.com/kuangliu/pytorch-ssd
        Transform between (bboxes, lables) <-> SSD output

        dboxes: default boxes in size 8732 x 4,
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format

        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes

        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        #print("# Bounding boxes: {}".format(self.nboxes))
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

        self.dboxes = self.dboxes.cpu()
        self.dboxes_xywh = self.dboxes_xywh.cpu()
        self.dboxes_npu = self.dboxes.npu()
        self.dboxes_xywh_npu = self.dboxes_xywh.npu()
    def encode(self, bboxes_in, labels_in, criteria = 0.5):

        # try: 
        # print("bboxes_in.shape, self.dboxes.shape", bboxes_in.shape, self.dboxes.shape)
        ious = calc_iou_tensor(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.long).cpu()   


        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()                     
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        # Transform format to xywh format
        x, y, w, h = 0.5*(bboxes_out[:, 0] + bboxes_out[:, 2]), \
                    0.5*(bboxes_out[:, 1] + bboxes_out[:, 3]), \
                    -bboxes_out[:, 0] + bboxes_out[:, 2], \
                    -bboxes_out[:, 1] + bboxes_out[:, 3]
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        # except:
        #     labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        #     bboxes_out = torch.zeros(self.nboxes, 4)
        # print("bboxes_out.shape, labels_out.shape", bboxes_out.shape, labels_out.shape)
        return bboxes_out, labels_out

    def scale_back_batch(self, bboxes_in, scores_in):
        """
            Do scale and transform from xywh to ltrb
            suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy*bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh*bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*self.dboxes_xywh_npu[:, :, 2:] + self.dboxes_xywh_npu[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp()*self.dboxes_xywh_npu[:, :, 2:]

        # Transform format to ltrb
        l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                     bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in,  criteria = 0.45, max_output=200):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
        N,A,C = probs.shape
        bboxes = bboxes.unsqueeze(-2).repeat([1,1,80,1])
        probs = probs[...,1:]
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(bboxes.half(), probs.half(),
                                                                                  0.05, criteria,
                                                                                  max_output, max_output)


        return nmsed_boxes, nmsed_classes, nmsed_scores


    # perform non-maximum suppression###############################################################
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
    #     # Reference to https://github.com/amdegroot/ssd.pytorch
        scores_in = scores_in.transpose(1, 0)
        #
        #     # Sort every row (in hopefully a single kernel launch)
        #     # NOTE: Not masked out things yet
        #     # NOTE: descending sort is easier to reason about
        #     # NOTE: Indices are to _global_ bboxes, we're not going to mask them
        score_sorted, score_sorted_idx = scores_in.sort(dim=1, descending=True)
        
        #     # Now generate the mask on the sorted scores
        mask = score_sorted > 0.05
        
        #     # number of default boxes per class that have a score > 0.05
        splits = mask.sum(dim=1).tolist()
        
        # only keep scores & indices for default boxes that contribute to this class
        # NOTE: Not masking out bboxes, all indices are global
        score_sorted = score_sorted[mask].split(splits)
        score_sorted_idx = score_sorted_idx[mask].split(splits)
        
        # assemble prefix sum of splits
        offsets = torch.tensor([0] + list(itertools.accumulate(splits)), dtype=torch.int32, device=bboxes_in.device)
        bboxes_out, scores_out, labels_out = C.nms(1, # N
                                                  scores_in.shape[0],
                                                  offsets,
                                                  torch.cat(score_sorted),
                                                  torch.cat(score_sorted_idx),
                                                  bboxes_in.contiguous(), # VITAL otherwise we get bad results :(
                                                  criteria,
                                                  max_num)
        
        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]

def npu_multiclass_nms(multi_bboxes,
                       multi_scores,
                       score_thr=0.05,
                       nms_thr=0.5,
                       max_num=200,
                       max_output = 200,
                       score_factors=None):
    """NMS for multi-class bboxes using npu api.

    Origin implement from mmdetection is
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/post_processing/bbox_nms.py#L7

    This interface is similar to the original interface, but not exactly the same.

    Args:
        multi_bboxes (Tensor): shape (n, #class, 4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
            On npu, in order to keep the semantics unblocked, we will unify the dimensions
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold. In the original implementation, a dictionary of {"iou_threshold": 0.45}
            was passed, which is simplified here.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept; if there are less than max_num bboxes after NMS,
            the output will zero pad to max_num. On the NPU, the memory needs to be requested in advance,
            so the current max_num cannot be set to -1 at present
        score_factors (Tensor): The factors multiplied to scores before applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels are 0-based.
    """

    num_classes = multi_scores.size(1)-1
    num_boxes = multi_scores.size(0)
    #print(num_classes,num_boxes)                     #####################
    if score_factors is not None:
        multi_scores = multi_scores[:, :-1] * score_factors[:, None]
    else:
        multi_scores = multi_scores[:, :-1]
    multi_bboxes = multi_bboxes.reshape(1, num_boxes, multi_bboxes.numel() // 4 // num_boxes, 4)
    multi_scores = multi_scores.reshape(1, num_boxes, num_classes)

    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(multi_bboxes.half(),
                                                                                  multi_scores.half(),
                                                                                  score_thr, nms_thr,
                                                                                  max_num, max_num)
    print(nmsed_boxes.shape, nmsed_scores.shape, nmsed_classes.shape, nmsed_num.shape)
    nmsed_boxes = nmsed_boxes.reshape(nmsed_boxes.shape[1:])
    nmsed_scores = nmsed_scores.reshape(nmsed_scores.shape[1])
    nmsed_classes = nmsed_classes.reshape(nmsed_classes.shape[1])             
    #print(nmsed_boxes.shape,nmsed_scores.shape,nmsed_classes.shape)
    #return torch.cat([nmsed_boxes, nmsed_scores[:, None]], -1), nmsed_classes
    _, max_ids = nmsed_scores.sort(dim=0)
    max_ids = max_ids[-max_output:]
    #print(type(nmsed_classes))
    ones = torch.ones(200).npu()
    nmsed_classes = nmsed_classes + ones
    #print(nmsed_boxes[max_ids, :], nmsed_classes[max_ids], nmsed_scores[max_ids])
    #return nmsed_boxes, nmsed_classes, nmsed_scores
    return nmsed_boxes[max_ids, :], nmsed_classes[max_ids], nmsed_scores[max_ids]
             
def npu_batched_multiclass_nms(
        multi_bboxes,
        multi_scores,
        score_thr=0.05,
        nms_thr=0.5,
        max_num=200,
        score_factors=None):
    """NMS for batched multi-class bboxes using npu api.

    Origin implement from mmdetection is
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/post_processing/bbox_nms.py#L7

    This interface is similar to the original interface, but not exactly the same.
    This interface implements the nms method under batch.

    Args:
        multi_bboxes (Tensor): shape (bs, n, #class, 4) or (bs, n, 4)
        multi_scores (Tensor): shape (bs, n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
            On npu, in order to keep the semantics unblocked, we will unify the dimensions
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold. In the original implementation, a dictionary of {"iou_threshold": 0.45}
            was passed, which is simplified here.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept; if there are less than max_num bboxes after NMS,
            the output will zero pad to max_num. On the NPU, the memory needs to be requested in advance,
            so the current max_num cannot be set to -1 at present
        score_factors (Tensor): The factors multiplied to scores before applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (bs, k, 5) and (bs, k, 1). Labels are 0-based.
    """

    num_classes = multi_scores.size(2) - 1
    num_boxes = multi_scores.size(1)
    batch_size = multi_scores.size(0)
    if score_factors is not None:
        multi_scores = multi_scores[..., :-1] * score_factors[..., None]
    else:
        multi_scores = multi_scores[..., :-1]
    multi_bboxes = multi_bboxes.reshape(batch_size, num_boxes, multi_bboxes.numel() // 4 // num_boxes // batch_size, 4)
    multi_scores = multi_scores.reshape(batch_size, num_boxes, num_classes)

    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(multi_bboxes.half(),
                                                                                  multi_scores.half(),
                                                                                  score_thr, nms_thr,
                                                                                  max_num, max_num)

    return torch.cat([nmsed_boxes, nmsed_scores[..., None]], -1), nmsed_classes, nmsed_scores

def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def build_ssd300_coder():
    return Encoder(dboxes300_coco())
