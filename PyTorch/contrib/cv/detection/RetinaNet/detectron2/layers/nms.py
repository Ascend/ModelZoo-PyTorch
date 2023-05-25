# -*- coding: utf-8 -*-
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

from typing import List
import torch
if torch.__version__ >= '1.8':
    import torch_npu
#from torchvision.ops import boxes as box_ops
#from torchvision.ops import nms  # BC-compat
import numpy as np

def npu_multiclass_nms(multi_bboxes,
                       multi_scores,
                       score_thr=0.05,
                       nms_thr=0.45,
                       max_num=50,
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

    return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num


def npu_batched_multiclass_nms(
        multi_bboxes,
        multi_scores,
        score_thr=0.05,
        nms_thr=0.45,
        max_num=50,
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

    return torch.cat([nmsed_boxes, nmsed_scores[..., None]], -1), nmsed_classes

def batched_nms_npu(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU > iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    '''
    npu_nms_with_mask function detail
    box_scores	    tensor	（N,8），N为候选框个数，8为候选框坐标与置信度得分
    iou_threshold	float	IOU阈值
    selected_boxes	tensor	（N,5），返回过滤后并排过序的候选框，N为输出候选框个数，5为坐标与置信度得分
    selected_idx	tensor	排过序的box在输入box列表中的位置索引
    selected_mask	tensor	当前候选框是否可用的标志
    '''
    print("bnn1")
    # boxes_cpu=[i.cpu() for i in boxes]
    # scores_cpu=[i.cpu() for i in scores]
    print("boxes",boxes.shape,boxes)
    print("scores",scores.shape,scores)
    _, _, keep_mask = \
        torch_npu.npu_nms_with_mask(
            torch.cat([boxes, scores[..., None]], 1), iou_threshold)
    #keep_mask=keep_mask.npu()
    print("bnn2")
    return keep_mask


def batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    print("================box1===================")
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return batched_nms_npu(boxes, scores, idxs, iou_threshold)
        #return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)
    #nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = \
    #    torch_npu.npu_batch_nms(boxes, scores, 0.0, iou_threshold, 400, 400)
    
    #return torch.cat([nmsed_boxes, nmsed_scores[..., None]], -1), nmsed_classes
    print("================box2===================")
    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    print("================box3===================")
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    print("================box4===================")
    return keep


# Note: this function (nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
def nms_rotated(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the rotated boxes according
    to their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.

    Note that RotatedBox (5, 3, 4, 2, -90) covers exactly the same region as
    RotatedBox (5, 3, 4, 2, 90) does, and their IoU will be 1. However, they
    can be representing completely different objects in certain tasks, e.g., OCR.

    As for the question of whether rotated-NMS should treat them as faraway boxes
    even though their IOU is 1, it depends on the application and/or ground truth annotation.

    As an extreme example, consider a single character v and the square box around it.

    If the angle is 0 degree, the object (text) would be read as 'v';

    If the angle is 90 degrees, the object (text) would become '>';

    If the angle is 180 degrees, the object (text) would become '^';

    If the angle is 270/-90 degrees, the object (text) would become '<'

    All of these cases have IoU of 1 to each other, and rotated NMS that only
    uses IoU as criterion would only keep one of them with the highest score -
    which, practically, still makes sense in most cases because typically
    only one of theses orientations is the correct one. Also, it does not matter
    as much if the box is only used to classify the object (instead of transcribing
    them with a sequential OCR recognition model) later.

    On the other hand, when we use IoU to filter proposals that are close to the
    ground truth during training, we should definitely take the angle into account if
    we know the ground truth is labeled with the strictly correct orientation (as in,
    upside-down words are annotated with -180 degrees even though they can be covered
    with a 0/90/-90 degree box, etc.)

    The way the original dataset is annotated also matters. For example, if the dataset
    is a 4-point polygon dataset that does not enforce ordering of vertices/orientation,
    we can estimate a minimum rotated bounding box to this polygon, but there's no way
    we can tell the correct angle with 100% confidence (as shown above, there could be 4 different
    rotated boxes, with angles differed by 90 degrees to each other, covering the exactly
    same region). In that case we have to just use IoU to determine the box
    proximity (as many detection benchmarks (even for text) do) unless there're other
    assumptions we can make (like width is always larger than height, or the object is not
    rotated by more than 90 degrees CCW/CW, etc.)

    In summary, not considering angles in rotated NMS seems to be a good option for now,
    but we should be aware of its implications.

    Args:
        boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
           (x_center, y_center, width, height, angle_degrees) format.
        scores (Tensor[N]): Scores for each one of the rotated boxes
        iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
        by Rotated NMS, sorted in decreasing order of scores
    """
    from detectron2 import _C

    return _C.nms_rotated(boxes, scores, iou_threshold)


# Note: this function (batched_nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
def batched_nms_rotated(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        iou_threshold (float):
           discards all overlapping boxes
           with IoU < iou_threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    assert boxes.shape[-1] == 5

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # Strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # Note that batched_nms in torchvision/ops/boxes.py only uses max_coordinate,
    # which won't handle negative coordinates correctly.
    # Here by using min_coordinate we can make sure the negative coordinates are
    # correctly handled.
    max_coordinate = (
        torch.max(boxes[:, 0], boxes[:, 1]) + torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).max()
    min_coordinate = (
        torch.min(boxes[:, 0], boxes[:, 1]) - torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).min()
    offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone()  # avoid modifying the original values in boxes
    boxes_for_nms[:, :2] += offsets[:, None]
    keep = nms_rotated(boxes_for_nms, scores, iou_threshold)
    return keep
