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
import os
from configparser import ConfigParser

import sys
import torch
import torch.onnx
import torch.nn as nn
from functools import partial
from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model

NB_CLS = 1000
DROP = 0.0
DROP_PATH = 0.1

config = ConfigParser()
config.read(filenames='url.ini',encoding = 'UTF-8')
value = config.get(section="DEFAULT", option="data")

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=str(value),
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

def pth2onnx(model_name="deit_small_patch16_224-cd65a155.pth", output_file=None):
    
    model = create_model(
        "deit_small_patch16_224",
        pretrained=False,
        num_classes=NB_CLS,
        drop_rate=DROP,
        drop_path_rate=DROP_PATH,
        drop_block_rate=None,
    )
    checkpoint = torch.load(model_name, map_location='cpu')

    model.load_state_dict(checkpoint['model'])
    model.eval()

    input_names = ["image"]
    output_names = ["class"]
    dummy_input = torch.randn(1, 3, 224, 224)
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}

    torch.onnx.export(model, dummy_input, output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=True)

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    pth2onnx(model_name=input_file, output_file=output_file)
