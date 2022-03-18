# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        if cfg.MODEL.DEVICE == "cuda":
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=True, use_npu=False)     # new add by luo
        elif "npu" in cfg.MODEL.DEVICE:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=False, use_npu=True)
        else:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=False, use_npu=False)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        if cfg.MODEL.DEVICE == "cuda":
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True, use_npu=False)  # center loss
        elif "npu" in cfg.MODEL.DEVICE:
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False, use_npu=True)  # center loss
        else:
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False, use_npu=False)  # center loss
        

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        if cfg.MODEL.DEVICE == "cuda":
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True, use_npu=False)  # center loss
        elif "npu" in cfg.MODEL.DEVICE:
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False, use_npu=True)  # center loss
        else:
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False, use_npu=False)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        if cfg.MODEL.DEVICE == "cuda":
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=True, use_npu=False)     # new add by luo
        elif "npu" in cfg.MODEL.DEVICE:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=False, use_npu=True)     # new add by luo
        else:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=False, use_npu=False)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                a = xent(score, target) 
                b = triplet(feat, target)[0]
                c = cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion


def make_loss_only_center(cfg, num_classes):
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        if cfg.MODEL.DEVICE == "cuda":
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True, use_npu=False)  # center loss
        elif "npu" in cfg.MODEL.DEVICE:
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False, use_npu=True)  # center loss
        else:
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False, use_npu=False)  # center loss
        

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        if cfg.MODEL.DEVICE == "cuda":
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True, use_npu=False)  # center loss
        elif "npu" in cfg.MODEL.DEVICE:
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False, use_npu=True)  # center loss
        else:
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False, use_npu=False)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        if cfg.MODEL.DEVICE == "cuda":
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=True, use_npu=False)     # new add by luo
        elif "npu" in cfg.MODEL.DEVICE:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=False, use_npu=True)     # new add by luo
        else:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=False, use_npu=False)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                a = xent(score, target) 
                b = triplet(feat, target)[0]
                c = cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion