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


import torch
import torch.onnx
import argparse
import os
import sys
sys.path.append('./')


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Full path to the WaveGlow checkpoint file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory for the exported WaveGlow ONNX model')
    return parser


def export_onnx(parser, args):
    """CPU"""
    dev = torch.device("cpu")
    model = torch.load(args.input, map_location=torch.device('cpu'))['model']
    model.to(dev)
    model = model.remove_weightnorm(model)
    model.eval()

    onnx_path = os.path.join(args.output, "waveglow.onnx")
    mel = torch.randn(1, 80, 1)

    with torch.no_grad():
        model.infer(mel)
        model.forward = model.infer
        forward_input = mel
        opset_version = 13
        input_names = ['mel']
        dynamic_axes = {'mel': {2:'mel_seq'},
                        'output_audio': {1: 'audio_seq'}}
        torch.onnx.export(model, forward_input, onnx_path, input_names=input_names,
                        export_params=True, verbose=True, opset_version=opset_version, 
                        output_names=['output_audio'], dynamic_axes = dynamic_axes)


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch WaveGlow Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    export_onnx(parser, args)

if __name__ == '__main__':
    main()
    print("Onnx converted successfully!")

