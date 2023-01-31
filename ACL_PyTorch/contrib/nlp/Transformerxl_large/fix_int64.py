# Copyright 2022 Huawei Technologies Co., Ltd
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

# coding: utf-8
import sys
import numpy as np
from magiconnx import OnnxGraph


INT64 = 7
INT32 = 6
MAXINT32 = 2147483647
MININT32 = -2147483648


def insert_cast_node(graph, before_node, node_name, dtype=6):
    cast_node = graph.add_node(
        node_name,
        'Cast',
        {'to': dtype}
    )
    graph.insert_node(before_node, cast_node, mode='after')


def insert_cast_after_shape(graph):
    shape_nodes = graph.get_nodes("Shape")
    for node in shape_nodes:
        node_name = node.name
        insert_name = 'expand_after_{}'.format(node_name)
        insert_cast_node(graph, node_name, insert_name)


def transfer_constantofshape(graph):
    constant_nodes = graph.get_nodes("ConstantOfShape")
    for node in constant_nodes:
        try:
            if node.attrs['value'].data_type == INT64:
                node.attrs['value'].data_type = INT32
        except:
            if node.attrs['value'].t.data_type == INT64:
                node.attrs['value'].t.data_type = INT32


def value_to_int32(node):
    node_value = node.value.copy()
    if (node_value > MAXINT32).any():
        node_value[node_value>MAXINT32] = MAXINT32
    if (node_value < MININT32).any():
        node_value[node_value<MININT32] = MININT32
    node.value = node_value.astype(np.int32)
    return node


def convert_all_constants(graph):
    constant_nodes = graph.get_nodes('Constant')
    for node in constant_nodes:
        if np.issubdtype(node.value.dtype, np.int64):
            node = value_to_int32(node)


def convert_all_initializers(graph):
    initializer_nodes = graph.get_nodes('Initializer')
    for node in initializer_nodes:
        if np.issubdtype(node.value.dtype, np.int64):
            node = value_to_int32(node)


def convert_cast_nodes(graph):
    cast_nodes = graph.get_nodes(op_type='Cast')
    for node in cast_nodes:
        if node['to'] == INT64:
            node._node.attribute[0].i = INT32
            # node.attrs['to'] = 6


def fix_int64(graph):
    convert_cast_nodes(graph)
    insert_cast_after_shape(graph)
    transfer_constantofshape(graph)
    convert_all_constants(graph)
    convert_all_initializers(graph)
    return graph

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    onnx_graph = OnnxGraph(input_path, rewrite_node=True)
    onnx_graph = fix_int64(onnx_graph)
    onnx_graph.save(output_path)
