# Copyright 2022 Huawei Technologies Co., Ltd
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
import torch
import torch.onnx
import torchvision
from collections import OrderedDict


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--model', default='densenet169', help='model')
    parser.add_argument('--device_id', default=0, type=int, help='device id')
    parser.add_argument('--pretrained_weight_path', default='./model_89.pth', help='pretrained weight path')
    parser.add_argument("--num_classes", default=1000, type=int, help='num of classes')
    args = parser.parse_args()
    return args


def proc_nodes_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert():
    model = torchvision.models.__dict__[args.model](num_classes=args.num_classes)
    print("model", model)
    checkpoint = torch.load(args.pretrained_weight_path, map_location='cpu')
    checkpoint = torch.load(args.pretrained_weight_path, map_location='cpu')
    checkpoint['model'] = proc_nodes_module(checkpoint, 'model')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "densenet169.onnx", input_names=input_names, output_names=output_names, opset_version=11)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f'npu:{args.device_id}')
    torch.npu.set_device(device)
    convert()