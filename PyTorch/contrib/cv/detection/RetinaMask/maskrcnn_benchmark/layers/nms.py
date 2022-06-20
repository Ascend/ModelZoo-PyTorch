# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from typing import List


def py_cpu_nms(boxes, scores, thresh):
    boxes = boxes.cpu()
    scores = scores.cpu()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = scores

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = torch.sort(scores, descending=True)

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.max(torch.tensor(0.0), xx2 - xx1 + 1)
        h = torch.max(torch.tensor(0.0), yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]

    res_keep = torch.tensor(keep)

    return res_keep


def batched_nms_npu(boxes, scores, iou_threshold):
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
        in decreasing order of scoresyong
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    _, _, keep_mask = \
        torch.npu_nms_with_mask(
            torch.cat([boxes, scores[..., None]], 1), iou_threshold)
    return keep_mask


def batched_nms(boxes, scores, idxs, iou_threshold):

    assert boxes.shape[-1] == 4
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return batched_nms_npu(boxes, scores, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for _id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == _id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


nms = py_cpu_nms
