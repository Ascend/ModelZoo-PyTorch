# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import torch
import torch.onnx
import torch._utils
from baseline.model import DeepMAR
from collections import OrderedDict


def proc_nodes_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]

        new_state_dict[name] = v
    return new_state_dict


def convert(batch_size, input_file, output_file):
    checkpoint = torch.load(input_file, map_location='cpu')
    checkpoint['state_dict'] = proc_nodes_module(checkpoint, 'state_dict')
    model = DeepMAR.DeepMAR_ResNet50()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    import onnx
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    torch.onnx.export(model, dummy_input, output_file, input_names=input_names, output_names=output_names,
                      opset_version=11, do_constant_folding=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="")
    parser.add_argument('--output_file', type=str, default="")
    parser.add_argument('--batch_size', type=int, default="")
    args = parser.parse_args()


    convert(args.batch_size, args.input_file, args.output_file)