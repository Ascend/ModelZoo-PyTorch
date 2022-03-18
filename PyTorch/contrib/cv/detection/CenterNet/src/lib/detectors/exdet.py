# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import torch.npu

from models.decode import exct_decode, agnex_ct_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform, transform_preds
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class ExdetDetector(BaseDetector):
  def __init__(self, opt):
    super(ExdetDetector, self).__init__(opt)
    self.decode = agnex_ct_decode if opt.agnostic_ex else exct_decode

  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.npu.synchronize() ###npu change
      output = self.model(images)[-1]
      t_heat = output['hm_t'].sigmoid_()
      l_heat = output['hm_l'].sigmoid_()
      b_heat = output['hm_b'].sigmoid_()
      r_heat = output['hm_r'].sigmoid_()
      c_heat = output['hm_c'].sigmoid_()
      torch.npu.synchronize() ###npu change
      forward_time = time.time()
      if self.opt.reg_offset:
        dets = self.decode(t_heat, l_heat, b_heat, r_heat, c_heat, 
                      output['reg_t'], output['reg_l'],
                      output['reg_b'], output['reg_r'], 
                      K=self.opt.K,
                      scores_thresh=self.opt.scores_thresh,
                      center_thresh=self.opt.center_thresh,
                      aggr_weight=self.opt.aggr_weight)
      else:
        dets = self.decode(t_heat, l_heat, b_heat, r_heat, c_heat, K=self.opt.K,
                      scores_thresh=self.opt.scores_thresh,
                      center_thresh=self.opt.center_thresh,
                      aggr_weight=self.opt.aggr_weight)
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      inp_height, inp_width = images.shape[2], images.shape[3]
      pred_hm = np.zeros((inp_height, inp_width, 3), dtype=np.uint8)
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      parts = ['t', 'l', 'b', 'r', 'c']
      for p in parts:
        tag = 'hm_{}'.format(p)
        pred = debugger.gen_colormap(
          output[tag][i].detach().cpu().numpy(), (inp_height, inp_width))
        if p != 'c':
          pred_hm = np.maximum(pred_hm, pred)
        else:
          debugger.add_blend_img(
            img, pred, 'pred_{}_{:.1f}'.format(p, scale))
      debugger.add_blend_img(img, pred_hm, 'pred_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_{:.1f}'.format(scale))
      for k in range(len(detection[i])):
        # print('detection', detection[i, k, 4], detection[i, k])
        if detection[i, k, 4] > 0.01:
          # print('detection', detection[i, k, 4], detection[i, k])
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_{:.1f}'.format(scale))

  def post_process(self, dets, meta, scale=1):
    out_width, out_height = meta['out_width'], meta['out_height']
    dets = dets.detach().cpu().numpy().reshape(2, -1, 14)
    dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
    dets = dets.reshape(1, -1, 14)
    dets[0, :, 0:2] = transform_preds(
      dets[0, :, 0:2], meta['c'], meta['s'], (out_width, out_height))
    dets[0, :, 2:4] = transform_preds(
      dets[0, :, 2:4], meta['c'], meta['s'], (out_width, out_height))
    dets[:, :, 0:4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    detections = np.concatenate(
        [detection for detection in detections], axis=0).astype(np.float32)
    classes = detections[..., -1]
    keep_inds = (detections[:, 4] > 0)
    detections = detections[keep_inds]
    classes = classes[keep_inds]

    results = {}
    for j in range(self.num_classes):
      keep_inds = (classes == j)
      results[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
      soft_nms(results[j + 1], Nt=0.5, method=2)
      results[j + 1] = results[j + 1][:, 0:5]

    scores = np.hstack([
      results[j][:, -1] 
      for j in range(1, self.num_classes + 1)
    ])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, -1] >= thresh)
        results[j] = results[j][keep_inds]
    return results


  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='exdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='exdet')
    debugger.show_all_imgs(pause=self.pause)
