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
from graph_fusion import GraphFusion
from auto_optimizer import OnnxGraph


def fix_cast(graph):
    for cast_node in graph.get_nodes(op_type="Cast"):
        if cast_node['to'] == 7:
            cast_node['to'] = 6


if __name__ == '__main__':
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    GraphFusion(input_model=input_model, output_model=output_model, opt_type=1)

    onnx_graph = OnnxGraph.parse(output_model)
    fix_cast(onnx_graph)
    onnx_graph.save(output_model)
