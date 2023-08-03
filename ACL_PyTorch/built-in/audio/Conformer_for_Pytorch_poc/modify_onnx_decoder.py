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
from graph_fusion import create_mask


def build_mask(graph, input_node, out_nodes):
    # build mask:
    # input_node -> build_mask -> add -> out_node1
    #                          -> add -> out_node2

    mask_end_node = create_mask(graph, input_node)
    unsqueeze_node = graph.add_node(
        "Unsqueeze_mask",
        "Unsqueeze",
        attrs={
            'axes': [1, 2]
        },
        inputs=mask_end_node.outputs,
        outputs=["out_Unsqueeze_mask"]
    )
    # reconnect
    for out_node in out_nodes:
        out_node.inputs.append("out_Unsqueeze_mask")


def get_mask_input(graph):
    inserted_add_list = []
    for softmax_node in graph.get_nodes(op_type="Softmax"):
        pre_node = graph.get_prev_node(softmax_node.inputs[0])
        if pre_node.op_type != "Add":
            # cross attention block
            inserted_add_node = graph.add_node(
                f"Add_before_{pre_node.name}",
                "Add"
            )
            graph.insert_node(softmax_node.name, inserted_add_node, mode='before')
            inserted_add_list.append(inserted_add_node)
    return inserted_add_list


if __name__ == '__main__':
    input_model = sys.argv[1]
    output_model = sys.argv[2]

    onnx_graph = OnnxGraph.parse(input_model)
    input_node = onnx_graph['memory']
    out_nodes = get_mask_input(onnx_graph)
    build_mask(onnx_graph, input_node, out_nodes)
    onnx_graph.save(output_model)
