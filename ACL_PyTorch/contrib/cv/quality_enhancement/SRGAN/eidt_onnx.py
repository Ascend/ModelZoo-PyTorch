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
from magiconnx import OnnxGraph
import sys

def fix(graph):
    nodes = graph.get_nodes(op_type='PRelu')
    for node in nodes:
        slope_node = graph[node.inputs[1]]
        slope_node_value = slope_node.value
        slope_node.value = np.tile(slope_node_value, (64, 1, 1))


if __name__ == '__main__':
    onnx_path = sys.argv[1]
    batch_size = sys.argv[2]
    onnx_graph = OnnxGraph(onnx_path)
    fix(onnx_graph)
    onnx_graph.save('srgan_fix_bs{}.onnx'.format(batch_size))