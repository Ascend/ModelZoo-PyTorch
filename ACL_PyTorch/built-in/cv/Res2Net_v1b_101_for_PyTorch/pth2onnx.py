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


import itertools
import argparse
from res2net_v1b import res2net101_v1b_26w_4s
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='res2net101_v1b inference')
parser.add_argument('-m', '--trained_model', default=None,
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-o', '--output', default=None,
                    type=str, help='ONNX model file')
args = parser.parse_args()

model = res2net101_v1b_26w_4s()
checkpoint = torch.load(args.trained_model, map_location=torch.device('cpu'))

presistent_buffers = {k: v for k, v in model.named_buffers() if k not in model._non_persistent_buffers_set}
local_name_params = itertools.chain(model.named_parameters(), presistent_buffers.items())
local_state = {k: v for k, v in local_name_params  if v is not None}

for name, param in checkpoint.items():
    if local_state[name].shape != param.shape:
        if 'conv1' in name or 'conv3' in name:
            n1, c1, h, w = local_state[name].shape
            n2, c2, h, w = param.shape
            if n1 == n2:
                c = (c1 - c2) // 4
                cell = c2 // 4
                checkpoint[name] = torch.cat([torch.cat((param[:, i * cell: (i + 1) * cell, ...],
                                                         torch.zeros(n1, c, h, w, dtype=param.dtype)),
                                                        1) for i in range(4)], 1)
            else:
                n = (n1 - n2) // 4
                cell = n2 // 4
                checkpoint[name] = torch.cat([torch.cat((param[i * cell: (i + 1) * cell, ...],
                                                         torch.zeros(n, c1, h, w, dtype=param.dtype)),
                                                        0) for i in range(4)], 0)
            elif 'bn1' in name or 'bn3' in name:
                cell = param.size(0) // 4
                n = (local_state[name].size(0) - param.size(0)) // 4
                checkpoint[name] = torch.cat([torch.cat((param[i * cell: (i + 1) * cell],
                                                         torch.zeros(n, dtype=param.dtype)),
                                                        0) for i in range(4)])
            else:
                if param.dim() == 1:
                    checkpoint[name] = torch.cat((param,
                                                  torch.zeros(local_state[name].size(0) - param.size(0), dtype=param.dtype)),
                                                 0)
                else:
                    n1, c1, h, w = local_state[name].shape
                    n2, c2, h, w = param.shape
                    param = torch.cat((param, torch.zeros(n2, c1 - c2, h, w, dtype=param.dtype)), 1)
                    checkpoint[name] = torch.cat((param, torch.zeros(n1 - n2, c1, h, w, dtype=param.dtype)), 0)

model.load_state_dict(checkpoint)
model.eval()

inputs = torch.rand(1, 3, 224, 224)
torch.onnx.export(model, inputs, args.output,
                  input_names=["x"], output_names=["output"],
                  dynamic_axes={"x": {0: "-1"}}, opset_version=11)
