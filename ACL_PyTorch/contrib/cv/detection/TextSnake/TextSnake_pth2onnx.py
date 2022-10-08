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

from network.textnet import TextNet
import torch
import argparse
import sys
sys.path.append('./TextSnake.pytorch')


def pth2onnx(args):
    device = torch.device('cpu')
    model = TextNet(is_training=False, backbone=args.net).to(device)
    state_dict = torch.load(args.input_file, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    # state_dict = {
    #     'lr': ,
    #     'epoch': ,
    #     'model': model.state_dict(),
    #     'optimizer':
    # for n, p in torch.load(args.input_file, map_location=lambda storage, loc: storage)['model'].items():
    #    if n in state_dict.keys():
    #        state_dict[n].copy_(p)
    #    else:
    #        raise KeyError(n)
    model.eval()
    model.to('cpu')

    input_names = ["image"]
    output_names = ["output"]
    dynamic_axes = {'image': {0: '-1'}, 'output': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model, dummy_input, args.output_file, input_names=input_names,
                      dynamic_axes=dynamic_axes, output_names=output_names, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--net', type=str, default='vgg')
    args = parser.parse_args()

    # input_file = './textsnake_vgg_180.pth'
    # output_file = './TextSnake.onnx'
    pth2onnx(args)
