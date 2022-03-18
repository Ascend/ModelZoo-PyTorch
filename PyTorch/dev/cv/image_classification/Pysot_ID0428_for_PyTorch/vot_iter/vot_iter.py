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

import sys
import cv2
import torch
import numpy as np
import os
from os.path import join

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from . import vot
from .vot import Rectangle, Polygon, Point


# modify root

cfg_root = "path/to/expr"
model_file = join(cfg_root, 'model.pth')
cfg_file = join(cfg_root, 'config.yaml')

def warmup(model):
    for i in range(10):
        #model.template(torch.FloatTensor(1,3,127,127).cuda())
        model.template(torch.FloatTensor(1,3,127,127).npu())

def setup_tracker():
    cfg.merge_from_file(cfg_file)

    model = ModelBuilder()
    #model = load_pretrain(model, model_file).cuda().eval()
    model = load_pretrain(model, model_file).npu().eval()

    tracker = build_tracker(model)
    warmup(model)
    return tracker


tracker = setup_tracker()

handle = vot.VOT("polygon")
region = handle.region()
try:
    region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                       region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
except:
    region = np.array(region)

cx, cy, w, h = get_axis_aligned_bbox(region)

image_file = handle.frame()
if not image_file:
    sys.exit(0)

im = cv2.imread(image_file)  # HxWxC
# init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
tracker.init(im, gt_bbox_)

while True:
    img_file = handle.frame()
    if not img_file:
        break
    im = cv2.imread(img_file)
    outputs = tracker.track(im)
    pred_bbox = outputs['bbox']
    result = Rectangle(*pred_bbox)
    score = outputs['best_score']
    if cfg.MASK.MASK:
        pred_bbox = outputs['polygon']
        result = Polygon(Point(x[0], x[1]) for x in pred_bbox)

    handle.report(result, score)
