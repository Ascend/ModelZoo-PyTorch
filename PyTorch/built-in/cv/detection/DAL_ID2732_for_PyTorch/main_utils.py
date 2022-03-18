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

import cv2
import numpy as np
import numpy.random as npr
import torch
import csv
from utils.utils import Normailize, Reshape
from torchvision.transforms import Compose
import ssl
from pathlib import Path


class Logger(object):

    def __init__(self, path, header):
        self.log_file = Path(path).open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def static_rescale(im, target_size_h, target_size_w, multiple=32):
    im_shape = im.shape
    im_size_h, im_size_w = im_shape[0:2]
    im_scale_h = float(target_size_h) / float(im_size_h)
    im_scale_w = float(target_size_w) / float(im_size_w)
    im_scale = min(im_scale_h, im_scale_w)
    im_scale_x = np.floor(im.shape[1] * im_scale / multiple) * multiple / im.shape[1]
    im_scale_y = np.floor(im.shape[0] * im_scale / multiple) * multiple / im.shape[0]
    im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)
    im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
    return im, im_scale


class StaticRescale(object):
    def __init__(self, target_size_h=600, target_size_w=600, multiple=32):
        self._target_size_h = target_size_h
        self._target_size_w = target_size_w
        self.multiple = multiple

    def __call__(self, im):
        im, im_scales = static_rescale(im, self._target_size_h, self._target_size_w, self.multiple)
        return im, im_scales


def static_single_scale_detect(model, src, target_size_h, target_size_w, conf=None):
    padded_ims = torch.zeros(1, 3, target_size_h, target_size_w)
    im, im_scales = StaticRescale(target_size_h, target_size_w)(src)
    height, width = im.shape[0], im.shape[1]
    padded_ims[0, :, :height, :width] = Compose([Normailize(), Reshape(unsqueeze=True)])(im)
    model, padded_ims = model.npu(), padded_ims.npu()
    with torch.no_grad():
        scores, classes, boxes = model(padded_ims, test_conf=conf)
    scores = scores.data.cpu().numpy()
    classes = classes.data.cpu().numpy()
    boxes = boxes.data.cpu().numpy()
    boxes[:, :4] = boxes[:, :4] / im_scales
    if boxes.shape[1] > 5:
        boxes[:, 5:9] = boxes[:, 5:9] / im_scales
    scores = np.reshape(scores, (-1, 1))
    classes = np.reshape(classes, (-1, 1))
    cls_dets = np.concatenate([classes, scores, boxes], axis=1)
    keep = np.where(classes > 0)[0]
    return cls_dets[keep, :]


def static_im_detect(model, src, target_size_h, target_size_w, conf=None):
    return static_single_scale_detect(model, src, target_size_h, target_size_w, conf=conf)


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """
    def __init__(self, weights=(10., 10., 10., 5., 15.)):
        self.weights = weights

    def encode(self, ex_rois, gt_rois):
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
        ex_widths = torch.clamp(ex_widths, min=1)
        ex_heights = torch.clamp(ex_heights, min=1)
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
        ex_thetas = ex_rois[:, 4]

        gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
        gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
        gt_widths = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)
        gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
        gt_thetas = gt_rois[:, 4]

        wx, wy, ww, wh, wt = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets_dt = wt * (torch.tan(gt_thetas / 180.0 * np.pi) - torch.tan(ex_thetas / 180.0 * np.pi))

        targets = torch.stack(
            (targets_dx, targets_dy, targets_dw, targets_dh, targets_dt), dim=1
        )
        return targets

    def decode(self, boxes, deltas, mode='xywht'):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        widths = torch.clamp(widths, min=1)
        heights = torch.clamp(heights, min=1)
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights
        thetas = boxes[:, :, 4]

        wx, wy, ww, wh, wt = self.weights
        dx = deltas[:, :, 0] / wx
        dy = deltas[:, :, 1] / wy
        dw = deltas[:, :, 2] / ww
        dh = deltas[:, :, 3] / wh
        dt = deltas[:, :, 4] / wt

        pred_ctr_x = ctr_x if 'x' not in mode else ctr_x + dx * widths
        pred_ctr_y = ctr_y if 'y' not in mode else ctr_y + dy * heights
        pred_w = widths if 'w' not in mode else torch.exp(dw) * widths
        pred_h = heights if 'h' not in mode else torch.exp(dh) * heights
        pred_t = thetas if 't' not in mode else torch.atan(torch.tan(thetas / 180.0 * np.pi) + dt) / np.pi * 180.0

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([
            pred_boxes_x1,
            pred_boxes_y1,
            pred_boxes_x2,
            pred_boxes_y2,
            pred_t], dim=2
        )
        return pred_boxes

    def batch_encode(self, batch_ex_rois, batch_gt_rois):
        batch_num, box_num, _ = batch_ex_rois.shape
        box_num_all = batch_num * box_num
        ex_rois = batch_ex_rois.reshape((box_num_all, -1))
        gt_rois = batch_gt_rois.reshape((box_num_all, -1))
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
        ex_widths = torch.clamp(ex_widths, min=1)
        ex_heights = torch.clamp(ex_heights, min=1)
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
        ex_thetas = ex_rois[:, 4]

        gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
        gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
        gt_widths = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)
        gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
        gt_thetas = gt_rois[:, 4]

        wx, wy, ww, wh, wt = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets_dt = wt * (torch.tan(gt_thetas / 180.0 * np.pi) - torch.tan(ex_thetas / 180.0 * np.pi))

        targets = torch.stack(
            (targets_dx, targets_dy, targets_dw, targets_dh, targets_dt), dim=1
        )
        batch_targets = targets.reshape((batch_num, box_num, -1))
        return batch_targets


class StaticCollector(object):
    """
    StaticCollector
    """

    def __init__(self, scales, multiple=32, max_num_boxes=None):
        if isinstance(scales, (int, float)):
            self.scales = np.array([scales, scales], dtype=np.int32)
        else:
            self.scales = np.array(scales, dtype=np.int32)
        self.multiple = multiple
        self.max_num_boxes = max_num_boxes

    def __call__(self, batch):
        target_size_h, target_size_w = self.scales
        target_size_h = int(np.floor(float(target_size_h) / self.multiple) * self.multiple)
        target_size_w = int(np.floor(float(target_size_w) / self.multiple) * self.multiple)
        rescale = StaticRescale(target_size_h, target_size_w, multiple=self.multiple)
        transform = Compose([Normailize(), Reshape(unsqueeze=False)])

        images = [sample['image'] for sample in batch]
        bboxes = [sample['boxes'] for sample in batch]
        batch_size = len(images)
        max_width, max_height = target_size_h, target_size_w
        num_params = bboxes[0].shape[-1]
        if self.max_num_boxes is None:
            max_num_boxes = max(bbox.shape[0] for bbox in bboxes)
        else:
            max_num_boxes = self.max_num_boxes
        padded_ims = torch.zeros(batch_size, 3, max_width, max_height)
        padded_boxes = torch.ones(batch_size, max_num_boxes, num_params) * -1
        for i in range(batch_size):
            im, bbox = images[i], bboxes[i]
            im, im_scale = rescale(im)
            height, width = im.shape[0], im.shape[1]
            padded_ims[i, :, :height, :width] = transform(im)
            if num_params < 9:
                bbox[:, :4] = bbox[:, :4] * im_scale
            else:
                bbox[:, :8] = bbox[:, :8] * np.hstack((im_scale, im_scale))
            padded_boxes[i, :bbox.shape[0], :] = torch.from_numpy(bbox)
        return {'image': padded_ims, 'boxes': padded_boxes}


def ssl_set():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
