# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the MIT License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class downsample_net(nn.Module):
    def __init__(self):
        super(downsample_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 3, 3, 1, 1)
        self.relu = nn.ReLU()
        weight_list = [
    [
        [
            [
                0.257
            ]
        ],
        [
            [
                0.504
            ]
        ],
        [
            [
                0.098
            ]
        ]
    ],
    
    [
        [
            [
                -0.148
            ]
        ],
        [
            [
                -0.291
            ]
        ],
        [
            [
                0.439

            ]
        ]
    ],
    
	[
        [
            [
                0.439
            ]
        ],
        [
            [
                -0.368
            ]
        ],
        [
            [
                -0.071
            ]
        ]
    ]
]
        bias_list = [0.0627450980392157, 0.5019607843137255, 0.5019607843137255]
        self.weight = Variable(torch.from_numpy(np.array(weight_list, dtype = 'float32')))
        self.bias = Variable(torch.from_numpy(np.array(bias_list, dtype = 'float32')))
        self.conv_define = nn.Conv2d(3, 3, 1, 1)
        self.conv_define.weight.data = self.weight
        self.conv_define.bias.data = self.bias

    def forward(self, x):
        x = self.relu(self.conv1(x))
        out = self.conv2(x)
        out = self.conv_define(out)
        out = torch.mul(out, 255.)
        return out


model_path = "/path/to/pkl"
dn = downsample_net()
dn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict = False)
print(dn)
dummy_input = torch.randn(1, 3, 1152, 1440)
torch.onnx.export(dn, dummy_input, "/path/to/onnx", verbose=True, keep_initializers_as_inputs=True)

