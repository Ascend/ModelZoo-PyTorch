# Copyright 2022 Huawei Technologies Co., Ltd
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


import sys
from collections import OrderedDict
import torch

sys.path.append(r"./pytorch-ssd")
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite


def pytorch2onnx(input_file, model_type, output_file):
    num_classes = 21
    model = create_mobilenetv2_ssd_lite(num_classes, 
                                        width_mult=1.0, is_test=True)
    if model_type =='base_net':
        pretrained_dic = torch.load(input_file, map_location='cpu')
    elif model_type =='trained':
        pretrained_dic = torch.load(input_file, 
                                    map_location='cpu')['state_dict']
    else:
        raise Exception('Unknown model type:', model_type)
    pretrained_dic = {k.replace('module.', ''): v 
                      for k, v in pretrained_dic.items()}
    model.load_state_dict(pretrained_dic)
    model.eval()

    dummy_input = torch.randn(1, 3, 300, 300)
    input_names = ["image"]
    output_names = ["scores", "boxes"]
    dynamic_axes = {
        'image': {0: '-1'}, 
        'scores': {0: '-1'}, 
        'boxes': {0: '-1'}
    }
    torch.onnx.export(model, dummy_input, output_file, 
                      input_names=input_names, 
                      output_names=output_names, 
                      dynamic_axes=dynamic_axes, 
                      opset_version=11, 
                      verbose=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Pytorch model convert to ONNX')
    parser.add_argument('--ckpt', default=None, 
                        help='input checkpoint file path')
    parser.add_argument('--onnx', default='out.onnx', 
                        help='output onnx file path')
    parser.add_argument('--type', default='base_net', 
                        help='choose a model type')    
    args = parser.parse_args()

    pytorch2onnx(args.ckpt, args.type, args.onnx)
