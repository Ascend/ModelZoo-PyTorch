# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
import numpy as np
from auto_optimizer import OnnxGraph
from auto_optimizer.graph_optimizer import GraphOptimizer
from auto_optimizer.pattern import KnowledgeFactory


def merge_add(graph):
    def get_next_node(node):
        return graph.get_next_nodes(node.outputs[0])[0]

    def check_next_node(node, op_type):
        next_node = get_next_node(node)
        return next_node.op_type == op_type

    def get_value_info(out_name):
        for value in graph.value_infos:
            if value.name == out_name:
                return value.shape
        return []

    for matmul_node in graph.get_nodes('MatMul'):
        # check pattern: matmul->add->reshape->add->reshape
        if not check_next_node(matmul_node, 'Add'):
            continue
        add_node1 = get_next_node(matmul_node)
        if not check_next_node(add_node1, 'Reshape'):
            continue
        reshape_node1 = get_next_node(add_node1)
        if not check_next_node(reshape_node1, 'Add'):
            continue
        add_node2 = get_next_node(reshape_node1)
        if not check_next_node(add_node2, 'Reshape'):
            continue
        reshape_node2 = get_next_node(add_node2)

        input_shape = get_value_info(graph[matmul_node.name].outputs[0])
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
        graph.remove(reshape_node1.name)
        graph.remove(add_node2.name)
        graph.remove(reshape_node2.name)


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
    original_graph = OnnxGraph.parse(args.input_path)
    model_name = os.path.basename(args.input_path)
    model_name = os.path.splitext(model_name)[0].replace('_sim', '')
    if model_name in {'swin_large_patch4_window12_384_bs16',
                      'swin_large_patch4_window12_384_bs32',
                      'swin_large_patch4_window12_384_bs64'}:
        original_graph.save(args.out_path)
    else:
        knowledges = KnowledgeFactory.get_knowledge_pool()
        optimizer = GraphOptimizer(list(knowledges.keys()))
        optimized_graph, _ = optimizer.apply_knowledges(original_graph)
        merge_add(optimized_graph)
        optimized_graph.save(args.out_path)
