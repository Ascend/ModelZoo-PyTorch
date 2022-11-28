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

import sys
sys.path.append(r'./efficientdet-pytorch')
import torch
from effdet.config import get_efficientdet_config
from effdet.efficientdet import EfficientDet
import argparse

parser = argparse.ArgumentParser(description='pth to onnx')

parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size 1/4/8/16/32')
parser.add_argument('--checkpoint', type=str, default='d0.pth', metavar='N',
                    help='pytorch checkpoint path ')
parser.add_argument('--out', type=str, default='d0.onnx', metavar='N',
                    help='export onnx model')

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_efficientdet_config(model_name='tf_efficientdet_d0')
    model = EfficientDet(config=config,pretrained_backbone=False)
    model_path = args.checkpoint
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    example = torch.randn(args.batch_size, 3, 512, 512)
    export_onnx_file = args.out
    torch.onnx.export(model, example, export_onnx_file, do_constant_folding=True, verbose=True, opset_version=11)