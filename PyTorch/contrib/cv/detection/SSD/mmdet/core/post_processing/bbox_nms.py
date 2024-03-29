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
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps


def multiclass_nms(multi_bboxes,
                       multi_scores,
                       score_thr,
                       nms_cfg,
                       max_num=50,
                       score_factors=None):
    """NMS for multi-class bboxes using npu api.

      This interface is similar to the original interface, but not exactly the same.

      Args:
          multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
          multi_scores (Tensor): shape (n, #class+1), where the last column
              contains scores of the background class, but this will be ignored.
              On NPU, the last dimension cannot be ignored, so it will be deleted in subsequent processing
          score_thr (float): bbox threshold, bboxes with scores lower than it
              will not be considered.
          nms_thr (float): NMS IoU threshold
          max_num (int): if there are more than max_num bboxes after NMS,
              only top max_num will be kept; if there are less than max_num bboxes after NMS,
              the output will zero pad to max_num. On the NPU, the memory needs to be requested in advance,
              so the current max_num cannot be set to -1 at present
          score_factors (Tensor): The factors multiplied to scores before applying NMS

      Returns:
          tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels are 0-based.
      """

    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if torch.onnx.is_in_onnx_export():
        if multi_bboxes.shape[1] > 4:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 4)
        scores = multi_scores[:, :-1]

        # filter out boxes with low scores
        valid_mask = scores > score_thr

        # We use masked_select for ONNX exporting purpose,
        # which is equivalent to bboxes = bboxes[valid_mask]
        # (TODO): as ONNX does not support repeat now,
        # we have to use this ugly code
        bboxes = torch.masked_select(
            bboxes,
            torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                        -1)).view(-1, 4)
        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = torch.masked_select(scores, valid_mask)
        labels = valid_mask.nonzero(as_tuple=False)[:, 1]

        if bboxes.numel() == 0:
            bboxes = multi_bboxes.new_zeros((0, 5))
            labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

            if torch.onnx.is_in_onnx_export():
                raise RuntimeError('[ONNX Error] Can not record NMS '
                                'as it has not been executed this time')
            return bboxes, labels

        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]

        return dets, labels[keep]
    else:
        num_boxes = multi_scores.size(0)
        if score_factors is not None:
            multi_scores = multi_scores[:, :-1] * score_factors[:, None]
        else:
            multi_scores = multi_scores[:, :-1]
        multi_bboxes = multi_bboxes.reshape(1, num_boxes, multi_bboxes.numel() // 4 // num_boxes, 4)
        multi_scores = multi_scores.reshape(1, num_boxes, num_classes)

        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(
            multi_bboxes.half(),
            multi_scores.half(),
            score_thr,
            nms_cfg['iou_threshold'],
            max_num,
            max_num
        )

        nmsed_boxes = nmsed_boxes.reshape(nmsed_boxes.shape[1:])
        nmsed_scores = nmsed_scores.reshape(nmsed_scores.shape[1])
        nmsed_classes = nmsed_classes.reshape(nmsed_classes.shape[1])

        return torch.cat([nmsed_boxes, nmsed_scores[:, None]], -1), nmsed_classes


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
