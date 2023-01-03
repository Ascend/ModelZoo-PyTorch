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


import numpy as np
from auto_optimizer import OnnxGraph
import argparse


parser = argparse.ArgumentParser(description='pth to onnx')
parser.add_argument('--model', type=str, default='d0_bs8_sim.onnx', metavar='N',
                    help='onnx model')
parser.add_argument('--out', type=str, default='d0_bs8_modify.onnx', metavar='N',

                    help='modified onnx')

args = parser.parse_args()
g = OnnxGraph.parse(args.model)
new_ini = g.add_initializer('new_ini', np.array(0).astype(np.float16))
node=g.get_nodes('Pad')[0]
g[node.inputs[2]] = new_ini
g.save(args.out)