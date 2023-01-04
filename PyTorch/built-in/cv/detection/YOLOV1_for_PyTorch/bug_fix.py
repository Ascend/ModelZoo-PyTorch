import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from utils import box_ops
from utils.misc import is_parallel
from models.yolo.yolov1 import YOLOv1


def create_grid(self, img_size):

    img_h = img_w = img_size
    fmp_h, fmp_w = img_h // self.stride[0], img_w // self.stride[0]
    grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).half().view(-1, 2)
    grid_xy = grid_xy.unsqueeze(0).to(self.device)

    return grid_xy

YOLOv1.create_grid = create_grid


def iou_score(bboxes_a, bboxes_b, batch_size):
    
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)


    area_i = torch.prod(br - tl, 1) * ((tl < br).all())
    iou = area_i / (area_a + area_b - area_i + 1e-14)

    return iou.view(batch_size, -1)

box_ops.iou_score = iou_score


def giou_score(bboxes_a, bboxes_b, batch_size):
    
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    area_i = torch.prod(br - tl, 1) * ((tl < br).all())
    area_u = area_a + area_b - area_i
    iou = (area_i / (area_u + 1e-14)).clamp(0)
    
    tl = torch.min(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.max(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_c = torch.prod(br - tl, 1) * ((tl < br).all())

    giou = (iou - (area_c - area_u) / (area_c + 1e-14))

    return giou.view(batch_size, -1)

box_ops.giou_score = giou_score


def ciou_score(bboxes_a, bboxes_b, batch_size):
    
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    area_i = torch.prod(br - tl, 1) * ((tl < br).all())
    iou = area_i / (area_a + area_b - area_i + 1e-7)

    cw = torch.max(bboxes_a[..., 2], bboxes_b[..., 2]) - torch.min(bboxes_a[..., 0], bboxes_b[..., 0])
    ch = torch.max(bboxes_a[..., 3], bboxes_b[..., 3]) - torch.min(bboxes_a[..., 1], bboxes_b[..., 1])

    c2 = cw ** 2 + ch ** 2 + 1e-7
    rho2 = ((bboxes_b[..., 0] + bboxes_b[..., 2] - bboxes_a[..., 0] - bboxes_a[..., 2]) ** 2 +
            (bboxes_b[..., 1] + bboxes_b[..., 3] - bboxes_a[..., 1] - bboxes_a[..., 3]) ** 2) / 4
    w1 = bboxes_a[..., 2] - bboxes_a[..., 0]
    h1 = bboxes_a[..., 3] - bboxes_a[..., 1]
    w2 = bboxes_b[..., 2] - bboxes_b[..., 0]
    h2 = bboxes_b[..., 3] - bboxes_b[..., 1]
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1. + 1e-7))

    ciou = iou - (rho2 / c2 + v * alpha)

    return ciou.view(batch_size, -1)

box_ops.ciou_score = ciou_score


class ModelEMA(object):
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.))
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.is_fused = False

    def update(self, model, x, model_params_fused=None):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            
            if x.device.type == "npu":
                from apex.contrib.combine_tensors import combine_npu
                d_inv = 1. - d
                d = torch.tensor([d], device=x.device)

                if not self.is_fused:
                    ema_all_params = []
                    for _, p in self.ema.named_parameters():
                        if p.dtype.is_floating_point:
                            ema_all_params.append(p)
                    self.ema_params_fused = combine_npu(ema_all_params)

                    ema_all_buffers = []
                    for _, b in self.ema.named_buffers():
                        if b.dtype.is_floating_point:
                            ema_all_buffers.append(b)
                        else:
                            continue
                    self.ema_buffers_fused = combine_npu(ema_all_buffers)

                    model_all_buffers = []
                    for _, b in model.named_buffers():
                        if b.dtype.is_floating_point:
                            model_all_buffers.append(b)
                        else:
                            continue
                    self.model_buffers_fused = combine_npu(model_all_buffers)

                    self.is_fused = True

                self.ema_params_fused *= d
                self.ema_params_fused.add_(model_params_fused, alpha=d_inv)

                self.ema_buffers_fused *= d
                self.ema_buffers_fused.add_(self.model_buffers_fused, alpha=d_inv)
            else:
                msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
                for k, v in self.ema.state_dict().items():
                    if v.dtype.is_floating_point:
                        v *= d
                        v += (1. - d) * msd[k].detach()