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
import sys
from magiconnx import OnnxGraph


def fix(graph):
    clip_node = graph['Clip_473']
    max_value = graph.add_initializer('max_value', np.array(999999999, dtype=np.float32))
    clip_node.inputs[2] = max_value.name


if __name__ == '__main__':
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    onnx_graph = OnnxGraph(input_model)
    fix(onnx_graph)
    onnx_graph.save(output_model)
