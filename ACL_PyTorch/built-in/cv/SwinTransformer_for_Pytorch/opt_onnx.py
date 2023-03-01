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


import os
import argparse
import numpy as np
from magiconnx import OnnxGraph
from magiconnx.optimize.optimizer_manager import OptimizerManager


def get_value_info(graph, out_name):
    for value in graph._model.graph.value_info:
        if value.name == out_name:
            return [dim.dim_value for dim in value.type.tensor_type.shape.dim]


def merge_add(graph):
    def check_next_node(node, op_type):
        try:
            next_node = graph.get_next_nodes(node.name)[0]
            if next_node.op_type != op_type:
                return False
        except:
            return False
        return True

    for matmul_node in graph.get_nodes(op_type="MatMul"):
        # check pattern: matmul->add->reshape->add->reshape
        if not check_next_node(matmul_node, "Add"):
            continue
        add_node1 = graph.get_next_nodes(matmul_node.name)[0]
        if not check_next_node(add_node1, "Reshape"):
            continue
        reshape_node1 = graph.get_next_nodes(add_node1.name)[0]
        if not check_next_node(reshape_node1, "Add"):
            continue
        add_node2 = graph.get_next_nodes(reshape_node1.name)[0]
        if not check_next_node(add_node2, "Reshape"):
            continue
        reshape_node2 = graph.get_next_nodes(add_node2.name)[0]

        input_shape = get_value_info(graph, graph[matmul_node.name].outputs[0])
        # merge add value
        add_value1 = graph[add_node1.inputs[1]].value
        add_value2 = graph[add_node2.inputs[1]].value
        broad_dim0 = input_shape[0]
        broad_dim1 = add_value2.shape[0]
        broad_dim2 = add_value2.shape[2]
        target_shape1 = graph[reshape_node1.inputs[1]].value
        add_value1 = np.tile(add_value1, (broad_dim0, 1, 1, 1))
        add_value2 = np.tile(add_value2, (target_shape1[0]//broad_dim1, 1, target_shape1[2]//broad_dim2, 1, 1))
        out_shape = add_value1.shape
        graph[add_node1.inputs[1]].value = add_value1 + add_value2.reshape(out_shape)

        # del reshape->add->reshape
        graph.del_node(reshape_node1.name)
        graph.del_node(add_node2.name)
        graph.del_node(reshape_node2.name)


def parse_arguments():
    parser = argparse.ArgumentParser(description='SwinTransformer preprocess.')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='input model path')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='output path for optimized model')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    onnx_graph = OnnxGraph(args.input_path)
    model_name = os.path.basename(args.input_path)
    model_name = os.path.splitext(model_name)[0].replace('_sim', '')
    if model_name in ["swin_large_patch4_window12_384_bs16",
                      "swin_large_patch4_window12_384_bs32",
                      "swin_large_patch4_window12_384_bs64"]:
        onnx_graph.save(args.out_path)
    else:
        optimize_manager_base = OptimizerManager(onnx_graph)
        optimize_manager_base.apply()
        merge_add(onnx_graph)
        onnx_graph.save(args.out_path)
