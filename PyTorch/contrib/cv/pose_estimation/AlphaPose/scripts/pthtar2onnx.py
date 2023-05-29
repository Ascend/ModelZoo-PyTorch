# Copyright 2020 Huawei Technologies Co., Ltd
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
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.onnx

from collections import OrderedDict

from alphapose.models import builder
from alphapose.opt import cfg, logger, opt

def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    '''
    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED),map_location=torch.device('cpu'))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()
    '''
    return model

def convert():
    model = preset_model(cfg)
    #checkpoint = torch.load("./pretrained_models/fast_res50_256x192.pth", map_location='cpu')
    #model.load_state_dict(checkpoint)
    model.eval()

    # preact.conv1
    # conv_out
    input_names = ["preact.conv1"]
    output_names = ["conv_out"]
    dummy_input = torch.randn(32, 3, 256, 192)
    torch.onnx.export(model, dummy_input, "alphapose_fastpose_npu.onnx", input_names=input_names, output_names=output_names,
                      opset_version=11)
    print('export onnx done!')

if __name__ == "__main__":
    convert()
# python3 ./scripts/pthtar2onnx.py --cfg ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml