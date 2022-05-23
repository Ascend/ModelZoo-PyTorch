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

import onnx
import torch
import argparse

from collections import OrderedDict
from convmixer_net import convmixer_1536_20


#============================================================================
# Functions
#============================================================================
def proc_nodes_module(checkpoint, device, atterName=None):
    new_state_dict = OrderedDict()
    checkpoint_items = checkpoint.items() if device == 'gpu' \
        else checkpoint[atterName].items()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def pth2onnx(input_file, output_file, device):
    checkpoint = torch.load(input_file, map_location=torch.device('cpu'))

    model = convmixer_1536_20()

    if device == 'gpu':
        checkpoint = proc_nodes_module(checkpoint, device)
        model.load_state_dict(checkpoint)
    else:
        checkpoint['state_dict'] = proc_nodes_module(checkpoint, device, 'state_dict')
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    input_names = ['image']
    output_names = ['class']
    dynamic_axes = {'image': {0: '-1'}, 'class':{0: '-1'}}

    dummy_img = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_img, output_file, dynamic_axes=dynamic_axes,
                      verbose=False, input_names=input_names, output_names=output_names, opset_version=11)


#============================================================================
# Main
#============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="./convmixer_1536_20_ks9_p7.pth.tar")
    parser.add_argument('--target', type=str, default="./convmixer_1536_20.onnx")
    parser.add_argument('--source-train-device', type=str, choices=['gpu', 'npu'], default='gpu', 
                        help="which training device the pth comes from")
    args = parser.parse_args()

    pth2onnx(args.source, args.target, args.source_train_device)
