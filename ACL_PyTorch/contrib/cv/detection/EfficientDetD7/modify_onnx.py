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
sys.path.append(r'onnx_tools/OXInterface')
from OXInterface import OXGraph
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='pth to onnx')
parser.add_argument('--model', type=str, default='d7.onnx', metavar='N',
                    help='onnx model')
parser.add_argument('--node', type=str, default='3080', metavar='N',
                    help='need to modify pad node number')
parser.add_argument('--out', type=str, default='d7_modify.onnx', metavar='N',
                    help='modified onnx')


args = parser.parse_args()
oxgraph = OXGraph(args.model)
oxinitializer_node = oxgraph.get_oxinitializer_by_name(args.node)
new_data = np.array(0, dtype=np.float32)
oxinitializer_node.set_data(new_data)
oxgraph.save_new_model(args.out)