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

import argparse
import torch
import lib.medzoo as medzoo
    
def pth2onnx():
    args = get_arguments()
    input_file = args.input
    output_file = args.output
    model, optimizer = medzoo.create_model(args)
    checkpoint = torch.load(input_file, map_location="cpu")
    model.load_state_dict(checkpoint, False)

    model.eval()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 4, 64, 64, 64)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, verbose=False, 
    opset_version=11)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--inChannels', type=int, default=4)
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--input', type=str, default='none')
    parser.add_argument('--output', type=str, default='none')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pth2onnx()