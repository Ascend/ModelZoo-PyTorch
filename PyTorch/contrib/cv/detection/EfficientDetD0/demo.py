# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import numpy as np
import time
import cv2
import torch.nn.parallel
import os
from effdet.bench import *
torch.backends.cudnn.benchmark = True
from effdet.config import get_efficientdet_config
import torch
from effdet.efficientdet import EfficientDet
import os
import logging
from collections import OrderedDict

def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                print(1)
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

_logger = logging.getLogger(__name__)

config = get_efficientdet_config(model_name='tf_efficientdet_d0')
config['soft_nms']=False
print(config)
eff_model = EfficientDet(config=config)

model_path = './output/train/checkpoint-243.pth.tar'
eff_model.load_state_dict(load_state_dict(model_path,use_ema=True)) #,map_location=torch.device('cpu')
eff_model = eff_model.npu()
eff_model.eval()
bench = DetBenchPredict(eff_model)
model_config = bench.config
param_count = sum([m.numel() for m in bench.parameters()])
print(param_count)
bench = bench.npu()
bench.eval()
img = cv2.imread('test.jpg')
torch_input=img.astype(np.float32)
# torch_input=cv2.resize(torch_input,(512,512))
torch_input = np.transpose(torch_input)
torch_input = np.expand_dims(torch_input, 0)
torch_input = torch.from_numpy(torch_input).npu()
predictions = []
with torch.no_grad():
    output = bench(torch_input, img_info=None)
    for img_det in output:
        predictions.append(img_det)
    coco_predictions = []
    for img_dets in predictions:
        if True:
            # to xyxy
            img_dets[:, 0:4] = img_dets[:, [1, 0, 3, 2]]
        # to xywh
        img_dets[:, 2] -= img_dets[:, 0]
        img_dets[:, 3] -= img_dets[:, 1]
        for det in img_dets:
            score = float(det[4])
            if score < .001:  # stop when below this threshold, scores in descending order
                break
            coco_det = dict(
                bbox=det[0:4].tolist(),
                score=score,
                category_id=int(det[5]))
            coco_predictions.append(coco_det)
            
print("coco_predictions",coco_predictions)
