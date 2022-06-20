# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
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
#
#
from detectron2.layers import batched_nms
import pdb

import torch

def ml_nms(boxlist, nms_thresh, max_proposals=-1,
           score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    if boxlist.has('pred_boxes'):
        boxes = boxlist.pred_boxes.tensor
        labels = boxlist.pred_classes
    else:
        boxes = boxlist.proposal_boxes.tensor
        labels = boxlist.proposal_boxes.tensor.new_zeros(
            len(boxlist.proposal_boxes.tensor))
    scores = boxlist.scores

    # pdb.set_trace()
    # print(boxes.size())
    # print(scores.size())
    # print('prepare')
    
    # keep = batched_nms(boxes, scores, labels, nms_thresh)
    nmsed_boxes, nmsed_scores, nmsed_classes, _ = batched_nms(boxes, scores, labels, nms_thresh)
    # if max_proposals > 0:
    #     keep = keep[: max_proposals]
    keep = torch.tensor(range(0, 400))
    boxlist = boxlist[keep]
    # pdb.set_trace()
    # print(keep.size())
    boxlist.pred_boxes.tensor = torch.squeeze(nmsed_boxes)
    boxlist.scores =  torch.squeeze(nmsed_scores)
    boxlist.pred_classes.tensor = torch.squeeze(nmsed_classes)
    return boxlist
