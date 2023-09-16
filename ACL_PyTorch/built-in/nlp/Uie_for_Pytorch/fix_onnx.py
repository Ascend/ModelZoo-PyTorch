# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import numpy as np

from auto_optimizer import OnnxGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode, Node, Initializer


def fix_mul(graph):
    for mul_node in graph.get_nodes(op_type='Mul'):
        prev_node = graph.get_prev_node(mul_node.inputs[0])
        next_node = graph.get_next_nodes(mul_node.outputs[0])[0]
        if not prev_node or prev_node.op_type != 'Transpose':
            continue
        node_num = int(mul_node.name.split(".")[-1])
        if len(mul_node.outputs) > 1:
            continue
        cast_node = mul_node.inputs[1]
        graph.remove(mul_node.name, {0: 0})
        new_mul = graph.add_node(f'p2o.Mul.{node_num}', 'Mul')
        graph.insert_node(next_node.name, new_mul)
        new_mul.inputs.append(cast_node)


def fix_add_shape(graph):
    first_add = graph['p2o.Add.6']
    reshape_before_add = graph.add_node(
        'reshape_before_add',
        'Reshape'
    )
    add_inti = graph.get_node(first_add.inputs[0], node_type=Initializer) or \
              graph.get_node(first_add.inputs[1], node_type=Initializer)
    if add_inti.value.shape[0] == 1:
        add_int_value = np.tile(graph[add_inti.name].value, (1, 1, 1)).reshape([-1, 768])
    else:
        add_int_value = graph[add_inti.name].value.reshape([-1, 768])
    graph[add_inti.name].value = add_int_value

    graph.insert_node(first_add.name, reshape_before_add, mode='before')
    reshape_init = graph.add_initializer(
        f"{reshape_before_add.name}_value",
        np.array([-1, 768], dtype='int64')
    )
    reshape_before_add.inputs.append(reshape_init.name)

    for softmax in graph.get_nodes(op_type='Softmax'):
        matmul_node = graph.get_next_nodes(softmax.outputs[0])[0]
        transpose_node = graph.get_next_nodes(matmul_node.outputs[0])[0]
        reshape_node = graph.get_next_nodes(transpose_node.outputs[0])[0]
        reshape_int = graph[reshape_node.inputs[1]]
        reshape_int.value = np.array([-1, 768], dtype='int64')


def fix_transpose(graph):
    for trans_node in graph.get_nodes(op_type='Transpose'):
        perm = trans_node.attrs.get('perm', [1])
        if perm[1] == 2 and perm[2] == 1:
            continue
        trans_node.attrs['perm'] = [0, 2, 1, 3]
        new_transpose = graph.add_node(
            name=f'{trans_node.name}_after',
            op_type='Transpose',
            attrs={'perm': [0, 1, 3, 2]}
        )
        graph.insert_node(trans_node.name, new_transpose, 0, mode='after')


def fix_reshape(graph, bs, seq_len):
    for reshape_node in graph.get_nodes(op_type='Reshape'):
        reshape_init = graph.get_node(reshape_node.inputs[0], node_type=Initializer) or \
                       graph.get_node(reshape_node.inputs[1], node_type=Initializer)
        if reshape_init.value[0] != 0:
            continue
        reshape_init.value = np.array([bs, seq_len, 12, 64], dtype='int64')

    final_add = graph['p2o.Add.346']
    reshape_before_add = graph.add_node(
        "reshape_before_final_add",
        "Reshape",
    )
    graph.insert_node(final_add.name, reshape_before_add, mode='before')
    reshape_init = graph.add_initializer(
        f'{reshape_before_add.name}_value',
        np.array([bs, seq_len, 768], dtype='int64')
    )
    reshape_before_add.inputs.append(reshape_init.name)

if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    bs_ = sys.argv[3]
    seq_len_ = sys.argv[4]
    onnx_graph = OnnxGraph.parse(input_path)
    fix_mul(onnx_graph)
    fix_add_shape(onnx_graph, bs_)
    fix_transpose(onnx_graph)
    fix_reshape(onnx_graph, bs_, seq_len_)
    onnx_graph.save(save_path)
