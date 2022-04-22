import numpy as np
import torch
from torch import nn

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from .retinanet_loss import make_retinanet_loss_evaluator
from .anchor_generator import make_anchor_generator_retinanet
from .retinanet_infer import make_retinanet_postprocessor
from .retinanet_detail_infer import make_retinanet_detail_postprocessor


class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.RETINANET.NUM_CLASSES - 1
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        num_anchors = len(cfg.RETINANET.ASPECT_RATIOS) \
                      * cfg.RETINANET.SCALES_PER_OCTAVE

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.RETINANET.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            if cfg.MODEL.USE_GN:
                cls_tower.append(nn.GroupNorm(32, in_channels))

            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            if cfg.MODEL.USE_GN:
                bbox_tower.append(nn.GroupNorm(32, in_channels))

            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )

        # Initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits,
                        self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
                if isinstance(l, nn.GroupNorm):
                    torch.nn.init.constant_(l.weight, 1.0)
                    torch.nn.init.constant_(l.bias, 0)

        # retinanet_bias_init
        prior_prob = cfg.RETINANET.PRIOR_PROB
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
        return logits, bbox_reg


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and RPN
    proposals and losses.
    """

    def __init__(self, cfg):
        super(RetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator_retinanet(cfg)
        head = RetinaNetHead(cfg)
        box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        if self.cfg.MODEL.SPARSE_MASK_ON:
            box_selector_test = make_retinanet_detail_postprocessor(
                cfg, 100, box_coder)
        else:
            box_selector_test = make_retinanet_postprocessor(
                cfg, 100, box_coder)
        box_selector_train = None
        if self.cfg.MODEL.MASK_ON or self.cfg.MODEL.SPARSE_MASK_ON:
            box_selector_train = make_retinanet_postprocessor(
                cfg, 100, box_coder)

        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test
        self.box_selector_train = box_selector_train
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression = self.head(features)

        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def permute_and_concat(self, list_tensor, channels):
        b, c = list_tensor[0].shape[:2]
        list_permute = [lt.permute(0, 2, 3, 1).reshape(b, -1, channels) for lt in list_tensor]
        return torch.cat(list_permute, 1)

    def _forward_train(self, anchors, box_cls, box_regression, targets):

        N = int(box_cls[0].size(0))
        A = int(box_regression[0].size(1) / 4)
        C = int(box_cls[0].size(1) / A)
        anchors_size = [anchor_list[0].size for anchor_list in anchors]
        anchors_bbox = [[anchor.bbox for anchor in anchor_list] for anchor_list in anchors]
        anchors_per_img = [torch.cat(anchor_list, 0) for anchor_list in anchors_bbox]

        box_cls = self.permute_and_concat(box_cls, C)
        box_regression = self.permute_and_concat(box_regression, 4)

        loss_box_cls, loss_box_reg = self.loss_evaluator(
            anchors_per_img, box_cls, box_regression, targets, C
        )
        losses = {
            "loss_retina_cls": loss_box_cls,
            "loss_retina_reg": loss_box_reg,
        }
        detections = None
        if self.cfg.MODEL.MASK_ON or self.cfg.MODEL.SPARSE_MASK_ON:
            with torch.no_grad():
                detections = self.box_selector_train(
                    anchors_per_img, box_cls, box_regression, anchors_size, N, C
                )

        return (anchors, detections), losses

    def _forward_test(self, anchors, box_cls, box_regression):
        N = int(box_cls[0].size(0))
        A = int(box_regression[0].size(1) / 4)
        C = int(box_cls[0].size(1) / A)
        anchors_size = [anchor_list[0].size for anchor_list in anchors]
        anchors_bbox = [[anchor.bbox for anchor in anchor_list] for anchor_list in anchors]
        anchors_per_img = [torch.cat(anchor_list, 0) for anchor_list in anchors_bbox]

        box_cls = self.permute_and_concat(box_cls, C)
        box_regression = self.permute_and_concat(box_regression, 4)
        boxes = self.box_selector_test(anchors_per_img, box_cls, box_regression, anchors_size, N, C)
        '''
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        '''
        return (anchors, boxes), {}


def build_retinanet(cfg):
    return RetinaNetModule(cfg)
