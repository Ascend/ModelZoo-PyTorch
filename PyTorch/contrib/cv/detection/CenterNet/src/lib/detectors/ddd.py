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

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import torch.npu ###

from models.decode import ddd_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ddd_post_process
from utils.debugger import Debugger
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

from .base_detector import BaseDetector

class DddDetector(BaseDetector):
  def __init__(self, opt):
    super(DddDetector, self).__init__(opt)
    self.calib = np.array([[707.0493, 0, 604.0814, 45.75831],
                           [0, 707.0493, 180.5066, -0.3454157],
                           [0, 0, 1., 0.004981016]], dtype=np.float32)


  def pre_process(self, image, scale, calib=None):
    height, width = image.shape[0:2]
    
    inp_height, inp_width = self.opt.input_h, self.opt.input_w
    c = np.array([width / 2, height / 2], dtype=np.float32)
    if self.opt.keep_res:
      s = np.array([inp_width, inp_height], dtype=np.int32)
    else:
      s = np.array([width, height], dtype=np.int32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = image #cv2.resize(image, (width, height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = (inp_image.astype(np.float32) / 255.)
    inp_image = (inp_image - self.mean) / self.std
    images = inp_image.transpose(2, 0, 1)[np.newaxis, ...]
    calib = np.array(calib, dtype=np.float32) if calib is not None \
            else self.calib
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio,
            'calib': calib}
    return images, meta
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.npu.synchronize() ###npu change
      output = self.model(images)[-1]
      output['hm'] = output['hm'].sigmoid_()
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      wh = output['wh'] if self.opt.reg_bbox else None
      reg = output['reg'] if self.opt.reg_offset else None
      torch.npu.synchronize() ###npu change
      forward_time = time.time()
      
      dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], wh=wh, reg=reg, K=self.opt.K)
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    detections = ddd_post_process(
      dets.copy(), [meta['c']], [meta['s']], [meta['calib']], self.opt)
    self.this_calib = meta['calib']
    return detections[0]

  def merge_outputs(self, detections):
    results = detections[0]
    for j in range(1, self.num_classes + 1):
      if len(results[j] > 0):
        keep_inds = (results[j][:, -1] > self.opt.peak_thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    dets = dets.detach().cpu().numpy()
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = ((img * self.std + self.mean) * 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    debugger.add_ct_detection(
      img, dets[0], show_box=self.opt.reg_bbox, 
      center_thresh=self.opt.vis_thresh, img_id='det_pred')
  
  def show_results(self, debugger, image, results):
    debugger.add_3d_detection(
      image, results, self.this_calib,
      center_thresh=self.opt.vis_thresh, img_id='add_pred')
    debugger.add_bird_view(
      results, center_thresh=self.opt.vis_thresh, img_id='bird_pred')
    debugger.show_all_imgs(pause=self.pause)