# Copyright 2023 Huawei Technologies Co., Ltd
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

# -*- coding:utf-8 -*-
import sys
from auto_optimizer import OnnxGraph


def keep_dymamic_batch(graph):
    for input_node in graph.inputs:
        input_node.shape[0] = 'batch'

    for out_node in graph.outputs:
        out_node.shape[0] = 'batch'

    graph.infershape()
    return graph


if __name__ == '__main__':
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    onnx_graph = OnnxGraph.parse(input_model)
    onnx_graph = keep_dymamic_batch(onnx_graph)
    onnx_graph.save(output_model)
