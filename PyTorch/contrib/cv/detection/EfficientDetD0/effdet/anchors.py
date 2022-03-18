
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

""" RetinaNet / EfficientDet Anchor Gen

Adapted for PyTorch from Tensorflow impl at
    https://github.com/google/automl/blob/6f6694cec1a48cdb33d5d1551a2d5db8ad227798/efficientdet/anchors.py

Hacked together by Ross Wightman, original copyright below
"""
# Copyright 2020 Google Research. All Rights Reserved.
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
# ==============================================================================
"""Anchor definition.

This module is borrowed from TPU RetinaNet implementation:
https://github.com/tensorflow/tpu/blob/master/models/official/retinanet/anchors.py
"""
from typing import Optional, Tuple, Sequence
#import torch.npu
import numpy as np
import torch
import torch.npu
import torch.nn as nn
#import torchvision.ops.boxes as tvb
#from torchvision.ops.boxes import batched_nms, remove_small_boxes 
from typing import List

from effdet.object_detection import ArgMaxMatcher, FasterRcnnBoxCoder, BoxList, IouSimilarity, TargetAssigner
from .soft_nms import batched_soft_nms

#from torchvision.ops import nms


# The minimum score to consider a logit for identifying detections.
MIN_CLASS_SCORE = -5.0

# The score for a dummy detection
_DUMMY_DETECTION_SCORE = -1e5


def decode_box_outputs(rel_codes, anchors, output_xyxy: bool=False):
    """Transforms relative regression coordinates to absolute positions.

    Network predictions are normalized and relative to a given anchor; this
    reverses the transformation and outputs absolute coordinates for the input image.

    Args:
        rel_codes: box regression targets.

        anchors: anchors on all feature levels.

    Returns:
        outputs: bounding boxes.

    """
    ycenter_a = (anchors[:, 0] + anchors[:, 2]) / 2
    xcenter_a = (anchors[:, 1] + anchors[:, 3]) / 2
    ha = anchors[:, 2] - anchors[:, 0]
    wa = anchors[:, 3] - anchors[:, 1]

    ty, tx, th, tw = rel_codes.unbind(dim=1)

    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    if output_xyxy:
        out = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    else:
        out = torch.stack([ymin, xmin, ymax, xmax], dim=1)
    return out


def clip_boxes_xyxy(boxes: torch.Tensor, size: torch.Tensor):
    boxes = boxes.clamp(min=0)
    size = torch.cat([size, size], dim=0)
    boxes = boxes.min(size.float())
    return boxes

def npu_multiclass_nms(multi_bboxes,
                       multi_scores,
                       score_thr:float=0.05,
                       nms_thr:float=0.45,
                       max_num:int=50,
                       score_factors:Optional[torch.Tensor]=None):
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

    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch.npu_batch_nms(multi_bboxes.half(), multi_scores.half(),
                                                                              score_thr, nms_thr,
                                                                              max_num, max_num)

    nmsed_boxes = nmsed_boxes.reshape(nmsed_boxes.shape[1:])
    nmsed_scores = nmsed_scores.reshape(nmsed_scores.shape[1])
    nmsed_classes = nmsed_classes.reshape(nmsed_classes.shape[1])

    return nmsed_boxes, nmsed_scores, nmsed_classes

def generate_detections(
        cls_outputs, box_outputs, anchor_boxes, indices, classes,
        img_scale: Optional[torch.Tensor], img_size: Optional[torch.Tensor],
        max_det_per_image: int = 100, soft_nms: bool = False):
    """Generates detections with RetinaNet model outputs and anchors.

    Args:
        cls_outputs: a torch tensor with shape [N, 1], which has the highest class
            scores on all feature levels. The N is the number of selected
            top-K total anchors on all levels.

        box_outputs: a torch tensor with shape [N, 4], which stacks box regression
            outputs on all feature levels. The N is the number of selected top-k
            total anchors on all levels.

        anchor_boxes: a torch tensor with shape [N, 4], which stacks anchors on all
            feature levels. The N is the number of selected top-k total anchors on all levels.

        indices: a torch tensor with shape [N], which is the indices from top-k selection.

        classes: a torch tensor with shape [N], which represents the class
            prediction on all selected anchors from top-k selection.

        img_scale: a float tensor representing the scale between original image
            and input image for the detector. It is used to rescale detections for
            evaluating with the original groundtruth annotations.

        max_det_per_image: an int constant, added as argument to make torchscript happy

    Returns:
        detections: detection results in a tensor with shape [max_det_per_image, 6],
            each row representing [x_min, y_min, x_max, y_max, score, class]
    """
    assert box_outputs.shape[-1] == 4
    assert anchor_boxes.shape[-1] == 4
    assert cls_outputs.shape[-1] == 1

    anchor_boxes = anchor_boxes[indices, :]

    # Appply bounding box regression to anchors, boxes are converted to xyxy
    # here since PyTorch NMS expects them in that form.
    boxes = decode_box_outputs(box_outputs.float(), anchor_boxes, output_xyxy=True)
    if img_scale is not None and img_size is not None:
        boxes = clip_boxes_xyxy(boxes, img_size / img_scale)  # clip before NMS better?

    scores = cls_outputs.sigmoid().squeeze(1).float()
    
    
    if soft_nms:
        top_detection_idx, soft_scores = batched_soft_nms(
            boxes, scores, classes, method_gaussian=True, iou_threshold=0.3, score_threshold=.001)
        scores[top_detection_idx] = soft_scores

        # keep only top max_det_per_image scoring predictions
        top_detection_idx = top_detection_idx[:max_det_per_image]
        boxes = boxes[top_detection_idx]
        scores = scores[top_detection_idx, None]
        classes = classes[top_detection_idx, None] + 1  # back to class idx with background class = 0
        
    else:
        
        device = boxes.device
        n = scores.shape[0]
        scores_zero = torch.zeros(n, 90+1, device=device,dtype=torch.float16)
        idx = torch.arange(0,n,device=device)
        scores_zero[idx.long(),classes] = scores
        boxes, scores, classes = npu_multiclass_nms(boxes, scores_zero, score_thr=0.005,
                       nms_thr=0.5,
                       max_num=max_det_per_image,
                       score_factors=None)

        classes = classes[:,None] + 1  # float
        scores = scores[:,None]

    if img_scale is not None:
        boxes = boxes * img_scale

    # FIXME add option to convert boxes back to yxyx? Otherwise must be handled downstream if
    # that is the preferred output format.

    # stack em and pad out to max_det_per_image if necessary
    
    num_det = max_det_per_image
    if soft_nms:
        detections = torch.cat([boxes, scores, classes.float()], dim=1)
    else:
        detections = torch.cat([boxes, scores, classes], dim=1)
    
    if num_det < max_det_per_image:
        detections = torch.cat([
            detections,
            torch.zeros((max_det_per_image - num_det, 6), device=detections.device, dtype=detections.dtype)
        ], dim=0)
    return detections


def get_feat_sizes(image_size: Tuple[int, int], max_level: int):
    """Get feat widths and heights for all levels.
    Args:
      image_size: a tuple (H, W)
      max_level: maximum feature level.
    Returns:
      feat_sizes: a list of tuples (height, width) for each level.
    """
    feat_size = image_size
    feat_sizes = [feat_size]
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append(feat_size)
    return feat_sizes


class Anchors(nn.Module):
    """RetinaNet Anchors class."""

    def __init__(self, min_level, max_level, num_scales, aspect_ratios, anchor_scale, image_size: Tuple[int, int]):
        """Constructs multiscale RetinaNet anchors.

        Args:
            min_level: integer number of minimum level of the output feature pyramid.

            max_level: integer number of maximum level of the output feature pyramid.

            num_scales: integer number representing intermediate scales added
                on each level. For instances, num_scales=2 adds two additional
                anchor scales [2^0, 2^0.5] on each level.

            aspect_ratios: list of tuples representing the aspect ratio anchors added
                on each level. For instances, aspect_ratios =
                [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.

            anchor_scale: float number representing the scale of size of the base
                anchor to the feature stride 2^level.

            image_size: Sequence specifying input image size of model (H, W).
                The image_size should be divided by the largest feature stride 2^max_level.
        """
        super(Anchors, self).__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        if isinstance(anchor_scale, Sequence):
            assert len(anchor_scale) == max_level - min_level + 1
            self.anchor_scales = anchor_scale
        else:
            self.anchor_scales = [anchor_scale] * (max_level - min_level + 1)

        assert isinstance(image_size, Sequence) and len(image_size) == 2
        # FIXME this restriction can likely be relaxed with some additional changes
        assert image_size[0] % 2 ** max_level == 0, 'Image size must be divisible by 2 ** max_level (128)'
        assert image_size[1] % 2 ** max_level == 0, 'Image size must be divisible by 2 ** max_level (128)'
        self.image_size = tuple(image_size)
        self.feat_sizes = get_feat_sizes(image_size, max_level)
        self.config = self._generate_configs()
        self.register_buffer('boxes', self._generate_boxes())

    @classmethod
    def from_config(cls, config):
        return cls(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)

    def _generate_configs(self):
        """Generate configurations of anchor boxes."""
        anchor_configs = {}
        feat_sizes = self.feat_sizes
        for level in range(self.min_level, self.max_level + 1):
            anchor_configs[level] = []
            for scale_octave in range(self.num_scales):
                for aspect in self.aspect_ratios:
                    anchor_configs[level].append(
                        ((feat_sizes[0][0] // feat_sizes[level][0],
                          feat_sizes[0][1] // feat_sizes[level][1]),
                         scale_octave / float(self.num_scales), aspect,
                         self.anchor_scales[level - self.min_level]))
        return anchor_configs

    def _generate_boxes(self):
        """Generates multiscale anchor boxes."""
        boxes_all = []
        for _, configs in self.config.items():
            boxes_level = []
            for config in configs:
                stride, octave_scale, aspect, anchor_scale = config
                base_anchor_size_x = anchor_scale * stride[1] * 2 ** octave_scale
                base_anchor_size_y = anchor_scale * stride[0] * 2 ** octave_scale
                if isinstance(aspect, Sequence):
                    aspect_x = aspect[0]
                    aspect_y = aspect[1]
                else:
                    aspect_x = np.sqrt(aspect)
                    aspect_y = 1.0 / aspect_x
                anchor_size_x_2 = base_anchor_size_x * aspect_x / 2.0
                anchor_size_y_2 = base_anchor_size_y * aspect_y / 2.0

                x = np.arange(stride[1] / 2, self.image_size[1], stride[1])
                y = np.arange(stride[0] / 2, self.image_size[0], stride[0])
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))

            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes).float()
        return anchor_boxes

    def get_anchors_per_location(self):
        return self.num_scales * len(self.aspect_ratios)


class AnchorLabeler(object):
    """Labeler for multiscale anchor boxes.
    """

    def __init__(self, anchors, num_classes: int, match_threshold: float = 0.5):
        """Constructs anchor labeler to assign labels to anchors.

        Args:
            anchors: an instance of class Anchors.

            num_classes: integer number representing number of classes in the dataset.

            match_threshold: float number between 0 and 1 representing the threshold
                to assign positive labels for anchors.
        """
        similarity_calc = IouSimilarity()
        matcher = ArgMaxMatcher(
            match_threshold,
            unmatched_threshold=match_threshold,
            negatives_lower_than_unmatched=True,
            force_match_for_each_row=True)
        box_coder = FasterRcnnBoxCoder()

        self.target_assigner = TargetAssigner(similarity_calc, matcher, box_coder)
        self.anchors = anchors
        self.match_threshold = match_threshold
        self.num_classes = num_classes
        self.indices_cache = {}

    def label_anchors(self, gt_boxes, gt_classes, filter_valid=True):
        """Labels anchors with ground truth inputs.

        Args:
            gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
                For each row, it stores [y0, x0, y1, x1] for four corners of a box.

            gt_classes: A integer tensor with shape [N, 1] representing groundtruth classes.

            filter_valid: Filter out any boxes w/ gt class <= -1 before assigning

        Returns:
            cls_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors]. The height_l and width_l
                represent the dimension of class logits at l-th level.

            box_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors * 4]. The height_l and
                width_l represent the dimension of bounding box regression output at l-th level.

            num_positives: scalar tensor storing number of positives in an image.
        """
        cls_targets_out = []
        box_targets_out = []

        if filter_valid:
            valid_idx = gt_classes > -1  # filter gt targets w/ label <= -1
            gt_boxes = gt_boxes[valid_idx]
            gt_classes = gt_classes[valid_idx]

        cls_targets, box_targets, matches = self.target_assigner.assign(
            BoxList(self.anchors.boxes), BoxList(gt_boxes), gt_classes)

        # class labels start from 1 and the background class = -1
        cls_targets = (cls_targets - 1).long()

        # Unpack labels.
        """Unpacks an array of cls/box into multiple scales."""
        count = 0
        for level in range(self.anchors.min_level, self.anchors.max_level + 1):
            feat_size = self.anchors.feat_sizes[level]
            steps = feat_size[0] * feat_size[1] * self.anchors.get_anchors_per_location()
            cls_targets_out.append(cls_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
            box_targets_out.append(box_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
            count += steps

        num_positives = (matches.match_results > -1).float().sum()

        return cls_targets_out, box_targets_out, num_positives

    def batch_label_anchors(self, gt_boxes, gt_classes, filter_valid=True):
        batch_size = len(gt_boxes)
        assert batch_size == len(gt_classes)
        num_levels = self.anchors.max_level - self.anchors.min_level + 1
        cls_targets_out = [[] for _ in range(num_levels)]
        box_targets_out = [[] for _ in range(num_levels)]
        num_positives_out = []

        anchor_box_list = BoxList(self.anchors.boxes)
        for i in range(batch_size):
            last_sample = i == batch_size - 1

            if filter_valid:
                valid_idx = gt_classes[i] > -1  # filter gt targets w/ label <= -1
                gt_box_list = BoxList(gt_boxes[i][valid_idx])
                gt_class_i = gt_classes[i][valid_idx]
            else:
                gt_box_list = BoxList(gt_boxes[i])
                gt_class_i = gt_classes[i]
            cls_targets, box_targets, matches = self.target_assigner.assign(anchor_box_list, gt_box_list, gt_class_i)

            # class labels start from 1 and the background class = -1
            cls_targets = (cls_targets - 1).long()

            # Unpack labels.
            """Unpacks an array of cls/box into multiple scales."""
            count = 0
            for level in range(self.anchors.min_level, self.anchors.max_level + 1):
                level_idx = level - self.anchors.min_level
                feat_size = self.anchors.feat_sizes[level]
                steps = feat_size[0] * feat_size[1] * self.anchors.get_anchors_per_location()
                cls_targets_out[level_idx].append(
                    cls_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
                box_targets_out[level_idx].append(
                    box_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
                count += steps
                if last_sample:
                    cls_targets_out[level_idx] = torch.stack(cls_targets_out[level_idx])
                    box_targets_out[level_idx] = torch.stack(box_targets_out[level_idx])

            num_positives_out.append((matches.match_results > -1).float().sum())
            if last_sample:
                num_positives_out = torch.stack(num_positives_out)

        return cls_targets_out, box_targets_out, num_positives_out

