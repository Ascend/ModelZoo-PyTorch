#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

import torch
if torch.__version__ >= '1.8':
    import torch_npu
from torch.autograd import Function
from ..box_utils import decode, nms, center_size
from data import voc_refinedet as cfg

def npu_multiclass_nms(multi_bboxes,
                       multi_scores,
                       score_thr=0.05,
                       nms_thr=0.45,
                       max_num=1000,
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

    num_classes = multi_scores.size(1) - 1
    num_boxes = multi_scores.size(0)
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

    nmsed_boxes = nmsed_boxes.reshape(nmsed_boxes.shape[1:])
    nmsed_scores = nmsed_scores.reshape(nmsed_scores.shape[1])
    nmsed_classes = nmsed_classes.reshape(nmsed_classes.shape[1])

    return torch.cat([nmsed_scores[:, None],nmsed_boxes], -1), nmsed_classes


class Detect_RefineDet(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh, 
                objectness_thre, keep_top_k):# 0, 1000, 0.05
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k            # 1000
        self.keep_top_k = keep_top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh 
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.objectness_thre = objectness_thre
        self.variance = cfg[str(size)]['variance']

    def forward(self, arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc_data = odm_loc_data
        conf_data = odm_conf_data

        arm_object_conf = arm_conf_data.data[:, :, 1:]
        no_object_index = arm_object_conf <= self.objectness_thre
        conf_data[no_object_index.expand_as(conf_data)] = 0

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            default = decode(arm_loc_data[i], prior_data, self.variance)
            default = center_size(default)
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            #print(decoded_boxes, conf_scores)
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                #print(scores.dim())
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)

                # ###
                # boxes = boxes.cpu()
                # scores = scores.cpu()
                # ids, count = nms(boxes, scores, self.nms_thresh, self.top_k) # 100

                # output[i, cl, :count] = \
                #         torch.cat((scores[ids[:count]].unsqueeze(1),
                #                boxes[ids[:count]]), 1).npu()    # score + bbox
                # print('cpu')
                # print( torch.cat((scores[ids[:count]].unsqueeze(1),
                #                boxes[ids[:count]]), 1))

                ###
                
                # boxes = boxes.npu()
                # scores = scores.npu()

                n = boxes.size(0)
                multi_bboxes = boxes.reshape(n, 1, 4)
                scores = scores.reshape(n, 1)
                pad_zeros = torch.zeros_like(scores)
                multi_scores = torch.cat((scores, pad_zeros), dim=-1)
                out, _ = npu_multiclass_nms(multi_bboxes, multi_scores)
                count = out.size(0)
                output[i, cl, :count] = out
                

        
        return output
