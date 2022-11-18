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
# limitations under the License.ls
import sys
sys.path.append(r'./detr')
from hubconf import detr_resnet50, detr_resnet50_onnx
import torch
import argparse

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
parser.add_argument('--batch_size', default=1,type=int)
args = parser.parse_args()

print(args)
mask = torch.zeros([args.batch_size, int(1280/32), int(1280/32)], dtype=torch.bool)
model = detr_resnet50_onnx(pretrained=False)
x = torch.rand(args.batch_size, 3, 1280, 1280)
inputs = [(x,mask)]
model.load_state_dict(torch.load('model/detr.pth', map_location="cpu")['model'])
model.eval()
torch.onnx.export(model, inputs[0], 'model/detr_bs{}.onnx'.format(args.batch_size), opset_version=11,
                  dynamic_axes={'inputs': { 2: '-1', 3: '-1'},
                                'mask': {1: '-1', 2: '-1'},
                                'pred_logits': {1:'-1',2:'-1'},
                                'pred_boxes': {1:'-1',2:'-1'}},
                  input_names=["inputs", "mask"],
                  output_names=["pred_logits", "pred_boxes"], verbose=True,
                  do_constant_folding=False
                  )