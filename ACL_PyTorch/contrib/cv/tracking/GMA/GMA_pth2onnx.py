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
import torch
import onnx
from collections import OrderedDict
sys.path.append('./GMA/core')
from network import RAFTGMA
import argparse




def pth2onnx(input_file, output_file, args, opset_version=12):
    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(input_file, map_location=torch.device('cpu')))
    model = model.module
    model.eval()
    dummy_input_0 = torch.randn(args.batchsize, 3, args.image_size[0], args.image_size[1]) # 0 dimension --> bs
    dummy_input_1 = torch.randn(args.batchsize, 3, args.image_size[0], args.image_size[1]) # 0 dimension --> bs
    dummy_input = (dummy_input_0, dummy_input_1)
    input_names = ['image1', 'image2']
    output_names = ['out1']
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, \
                      output_names = output_names, opset_version=opset_version, verbose=True) 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[440, 1024])
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--output_file', type=str, default="gma.onnx", help='.onnx output file')
    parser.add_argument('--input_file', type=str, help='.pth input file')
    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--batchsize', type=int)

    args = parser.parse_args()
    output_file = args.output_file
    input_file = args.input_file
    pth2onnx(input_file=input_file, output_file=output_file, args=args, opset_version=12)