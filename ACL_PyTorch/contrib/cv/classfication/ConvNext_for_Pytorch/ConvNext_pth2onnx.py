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
import torch
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
import sys
sys.path.append(r"./ConvNeXt")
from models import convnext
import os
from collections import OrderedDict

def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict
        
if __name__ == '__main__':
    model = create_model(
            'convnext_tiny', 
            pretrained=False, 
            num_classes=1000, 
            drop_path_rate=0.1,
            layer_scale_init_value=1e-06,
            head_init_scale=1.0,
            )

    pth = sys.argv[1]
    output_file = sys.argv[2]
    checkpoint = torch.load(os.path.join('./', pth), map_location='cpu')
    
    checkpoint = proc_nodes_module(checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()  

    input_names = ["image"]  
    output_names = ["class"]  
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}} 

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, 
    dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11, verbose=True) 

