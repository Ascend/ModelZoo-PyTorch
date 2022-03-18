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
import argparse
import sys
sys.path.append('./yolor')
from models.models import *

def pth2onnx(args):
    cfg, imgsz, weights = args.cfg, args.img_size, args.weights
    device = torch.device('cpu')
    model = Darknet(cfg, imgsz)
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    # model.load_state_dict(state_dict['model'])
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
    # model.to(torch.float16)
    model.to('cpu')
    print('load success')
    bs = args.batch_size
    bs_str = str(bs)
    input_names = ["image"]
    output_names = ["output"]
    dynamic_axes = {'image': {0: bs_str}, 'output': {0: '-1'}}
    dummy_input = torch.randn(bs, 3, 1344, 1344)
    torch.onnx.export(model, dummy_input, args.output_file, opset_version=11, input_names=input_names, output_names=output_names, verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    pth2onnx(args)
