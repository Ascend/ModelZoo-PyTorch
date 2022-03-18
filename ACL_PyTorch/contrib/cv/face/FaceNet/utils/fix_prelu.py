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
import numpy as np
from magiconnx import OnnxGraph


def fix_prelu(graph):
    prelu_nodes = graph.get_nodes(op_type='PRelu')
    for node in prelu_nodes:
        slope_para = graph[node.inputs[1]]
        fix_value = np.expand_dims(slope_para.value, axis=0)
        slope_para.value = fix_value
    return graph


if __name__ == '__main__':
    input_model = sys.argv[1]
    out_model = sys.argv[2]
    onnx_graph = OnnxGraph(input_model)
    onnx_graph = fix_prelu(onnx_graph)
    onnx_graph.save(out_model)
