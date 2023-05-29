# Copyright 2023 Huawei Technologies Co., Ltd
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

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes


def batched_nms(boxes, scores, max_output_size, iou_threshold, scores_threshold):
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
    num_classes = scores.size(1)
    num_boxes = scores.size(0)
    multi_bboxes = boxes.reshape(1, num_boxes, -1, 4)
    multi_scores = scores.reshape(1, num_boxes, num_classes)
    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(multi_bboxes.half(),
                                                                                  multi_scores.half(),
                                                                                  scores_threshold,
                                                                                  iou_threshold,
                                                                                  max_output_size,
                                                                                  max_output_size)
    nmsed_boxes = nmsed_boxes.reshape(nmsed_boxes.shape[1:])
    nmsed_scores = nmsed_scores.reshape(nmsed_scores.shape[1])
    nmsed_classes = nmsed_classes.reshape(nmsed_classes.shape[1])
    nmsed_num = nmsed_num.item()

    return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num


class RetinaNetPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(
            self,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            min_size,
            box_coder=None,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            box_coder (BoxCoder)
        """
        super(RetinaNetPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder

    def forward_for_single_feature_map(self, anchors, box_cls, box_regression,
                                       pre_nms_thresh):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = box_cls.device
        N, _, H, W = box_cls.shape
        A = int(box_regression.size(1) / 4)
        C = int(box_cls.size(1) / A)

        # put in the same format as anchors
        box_cls = box_cls.permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C)
        box_cls = box_cls.sigmoid().cpu().float()

        box_regression = box_regression.permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4).cpu().float()

        num_anchors = A * H * W

        results = [[] for _ in range(N)]
        candidate_inds = box_cls > pre_nms_thresh
        if candidate_inds.sum().item() == 0:
            empty_boxlists = []
            for a in anchors:
                empty_boxlist = BoxList(torch.zeros(1, 4).cpu().float(), a.size)
                empty_boxlist.add_field(
                    "labels", torch.LongTensor([-1]).cpu())
                empty_boxlist.add_field(
                    "scores", torch.Tensor([0]).cpu().float())
                empty_boxlists.append(empty_boxlist)
            return empty_boxlists

        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        for batch_idx, (per_box_cls, per_box_regression, per_pre_nms_top_n,
                        per_candidate_inds, per_anchors) in enumerate(
            zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors)):
            # Sort and select TopN
            per_box_cls = per_box_cls[per_candidate_inds]
            per_box_cls, top_k_indices = \
                per_box_cls.topk(per_pre_nms_top_n, sorted=False)

            per_candidate_nonzeros = \
                per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]
            per_class += 1

            detections = self.box_coder.decode_cpu(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results[batch_idx] = boxlist

        return results

    def forward(self, anchors_per_img, box_cls, box_regression, anchors_size, N, C, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        device = box_cls.device
        box_cls = box_cls.sigmoid()
        k = self.pre_nms_top_n * 2
        results = []
        for i in range(N):
            cls_scores = box_cls[i]
            bboxes = box_regression[i]
            achrs = anchors_per_img[i]
            anchor_size = anchors_size[i]
            bboxes = self.box_coder.decode(
                bboxes.view(-1, 4),
                achrs.view(-1, 4)
            )
            if not self.training:
                k = k * 2
                scores, topk_inds = torch.topk(cls_scores.flatten(), k=k, largest=True)
                labels = topk_inds % C
                topk_inds = topk_inds // C
                bboxes = bboxes[topk_inds]
            else:
                max_scores, labels = torch.max(cls_scores, 1)
                topk_scores, topk_inds = torch.topk(max_scores, k=k, largest=True)
                bboxes = bboxes[topk_inds]
                scores = topk_scores
                labels = labels[topk_inds]
            if labels.numel() == 0:
                result = BoxList(bboxes.new_ones([1, 4]), anchor_size, mode="xyxy")
                result.add_field("scores", bboxes.new_zeros([1, ]))
                result.add_field("labels", bboxes.new_full((1,), -1, dtype=torch.long))
            else:
                multi_scores = scores.new_zeros([k, C])
                multi_bboxes = bboxes.new_zeros([k, 4])
                k = min(k, labels.numel())
                multi_bboxes[:k] = bboxes[:k]
                indices = torch.arange(0, k).to(device)
                multi_scores[indices, labels[:k]] = scores[:k]

                nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = batched_nms(multi_bboxes, multi_scores,
                                                                                  self.fpn_post_nms_top_n,
                                                                                  iou_threshold=self.nms_thresh,
                                                                                  scores_threshold=self.pre_nms_thresh)
                nmsed_classes = nmsed_classes + 1
                result = BoxList(nmsed_boxes, anchor_size, mode="xyxy")
                result.add_field("scores", nmsed_scores)
                result.add_field("labels", nmsed_classes)
                result = result.clip_to_image(remove_empty=False)

            result.bbox = result.bbox.to(device)
            result.add_field('labels', result.get_field('labels').to(device))
            result.add_field('scores', result.get_field('scores').to(device))
            results.append(result)

        return results


def make_retinanet_postprocessor(
        config, fpn_post_nms_top_n, rpn_box_coder):
    pre_nms_thresh = 0.05
    pre_nms_top_n = config.RETINANET.PRE_NMS_TOP_N
    nms_thresh = 0.4
    fpn_post_nms_top_n = fpn_post_nms_top_n
    min_size = 0

    # nms_thresh = config.MODEL.RPN.NMS_THRESH
    # min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RetinaNetPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        box_coder=rpn_box_coder,
        min_size=min_size
    )
    return box_selector
