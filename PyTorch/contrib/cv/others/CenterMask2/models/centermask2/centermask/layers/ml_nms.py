# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from detectron2.layers import batched_nms
import pdb
import torch
from detectron2.layers import batched_nms_npu

# lzy 12.21 add nF
# from torch.contrib.npu.optimized_lib import function as nF
# end
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
   # lzy 12.20 turn tensor into cpu
    boxes = boxlist.pred_boxes.tensor.cpu()
    boxes_l = boxlist.pred_boxes.tensor

    scores = boxlist.scores.cpu()
    scores_l = boxlist.scores

   # end
    labels = boxlist.pred_classes

    keep = batched_nms(boxes, scores, labels, nms_thresh)
    

    if max_proposals > 0:
        keep = keep[: max_proposals].long()
    boxlist = boxlist[keep]
    return boxlist
