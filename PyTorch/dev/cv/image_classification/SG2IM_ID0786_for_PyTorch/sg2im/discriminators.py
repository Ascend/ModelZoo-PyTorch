#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from sg2im.bilinear import crop_bbox_batch
from sg2im.layers import GlobalAvgPool, Flatten, get_activation, build_cnn


class PatchDiscriminator(nn.Module):
  def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
               padding='same', pooling='avg', input_size=(128,128),
               layout_dim=0):
    super(PatchDiscriminator, self).__init__()
    input_dim = 3 + layout_dim
    arch = 'I%d,%s' % (input_dim, arch)
    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling,
      'padding': padding,
    }
    self.cnn, output_dim = build_cnn(**cnn_kwargs)
    self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)

  def forward(self, x, layout=None):
    if layout is not None:
      x = torch.cat([x, layout], dim=1)
    return self.cnn(x)


class AcDiscriminator(nn.Module):
  def __init__(self, vocab, arch, normalization='none', activation='relu',
               padding='same', pooling='avg'):
    super(AcDiscriminator, self).__init__()
    self.vocab = vocab

    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling, 
      'padding': padding,
    }
    cnn, D = build_cnn(**cnn_kwargs)
    self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
    num_objects = len(vocab['object_idx_to_name'])

    self.real_classifier = nn.Linear(1024, 1)
    self.obj_classifier = nn.Linear(1024, num_objects)

  def forward(self, x, y):
    if x.dim() == 3:
      x = x[:, None]
    vecs = self.cnn(x)
    real_scores = self.real_classifier(vecs)
    obj_scores = self.obj_classifier(vecs)
    ac_loss = F.cross_entropy(obj_scores, y)
    return real_scores, ac_loss


class AcCropDiscriminator(nn.Module):
  def __init__(self, vocab, arch, normalization='none', activation='relu',
               object_size=64, padding='same', pooling='avg'):
    super(AcCropDiscriminator, self).__init__()
    self.vocab = vocab
    self.discriminator = AcDiscriminator(vocab, arch, normalization,
                                         activation, padding, pooling)
    self.object_size = object_size

  def forward(self, imgs, objs, boxes, obj_to_img):
    crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
    real_scores, ac_loss = self.discriminator(crops, objs)
    return real_scores, ac_loss
