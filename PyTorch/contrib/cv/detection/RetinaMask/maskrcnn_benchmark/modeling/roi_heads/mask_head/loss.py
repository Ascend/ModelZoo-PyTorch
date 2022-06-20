# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


def project_masks_on_boxes(segmentation_masks, proposals, labels_per_img, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    num_targets = len(labels_per_img)
    proposals = proposals.bbox.to(torch.device("cpu")).float()
    masks = proposals.new_zeros([num_targets, M, M])
    proposals = proposals.numpy()
    labels = labels_per_img.to(torch.device("cpu"))

    for i, (segmentation_mask, proposal, label) in enumerate(zip(segmentation_masks, proposals, labels)):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        if label <= 0:
            continue
        mask = segmentation_mask.crop_and_resize_and_decode(proposal, (M, M))
        masks[i] = mask
    masks = masks.to(device)
    return masks


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size, fg_thr, bg_thr):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.fg_thr = fg_thr
        self.bg_thr = bg_thr

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_vals, matches = match_quality_matrix.max(dim=0)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matches]
        return matched_vals, matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_vals, matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_vals < self.fg_thr
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            segmentation_masks = matched_targets.get_field("masks")

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, proposals_per_image, labels_per_image, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)

        return labels, masks

    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = (labels > 0).half()

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        pos_cnt = positive_inds.sum()
        if pos_cnt == 0:
            return mask_logits.sum() * 0
        labels_inds = torch.arange(0, labels.size(0)).to(labels.device)
        positive_inds = positive_inds.view(-1, 1, 1).expand(-1, 28, 28)
        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[labels_inds, labels], mask_targets, weight=positive_inds, reduction='sum') / (
                            pos_cnt * 28 * 28)

        return mask_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION, cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD
    )

    return loss_evaluator
