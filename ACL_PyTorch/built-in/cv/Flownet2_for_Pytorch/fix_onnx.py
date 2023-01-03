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


import sys
import numpy as np
from magiconnx import OnnxGraph


INT64 = 7
INT32 = 6
MAXINT32 = 2147483647
MININT32 = -2147483648


def transfer_constantofshape(graph):
    constant_nodes = graph.get_nodes("ConstantOfShape")
    for node in constant_nodes:
        if node.attrs['value'].data_type == INT64:
            node.attrs['value'].data_type = INT32


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


def mergeIntializer(oxgraph, initializer1, initializer2, merged_name):
    """
    :param oxgraph: input onnx graph
    :param initializer1: initializer need to be merged
    :param initializer2: initializer need to be merged
    :return: merged initializer
    """
    merged_data = np.append(
        initializer1.value,
        initializer2.value,
    )
    merged_initializer = oxgraph.add_initializer(name=merged_name, value=merged_data)
    return merged_initializer


def merge_slice_node(oxgraph, slice_node1, slice_node2):
    # modify node1+node2 -> merge_node
    slice_node2.inputs[1] = mergeIntializer(
        oxgraph,
        oxgraph[slice_node1.inputs[1]],
        oxgraph[slice_node2.inputs[1]],
        '{}_1'.format(slice_node1.name)).name
    slice_node2.inputs[2] = mergeIntializer(
        oxgraph,
        oxgraph[slice_node1.inputs[2]],
        oxgraph[slice_node2.inputs[2]],
        '{}_2'.format(slice_node1.name)).name
    slice_node2.inputs[3] = mergeIntializer(
        oxgraph,
        oxgraph[slice_node1.inputs[3]],
        oxgraph[slice_node2.inputs[3]],
        '{}_3'.format(slice_node1.name)).name
    slice_node2.inputs[4] = mergeIntializer(
        oxgraph,
        oxgraph[slice_node1.inputs[4]],
        oxgraph[slice_node2.inputs[4]],
        '{}_4'.format(slice_node1.name)).name
    oxgraph.del_node(slice_node1.name)


def get_continuous_op(oxgraph, op_type='Slice'):
    all_ops = oxgraph.get_nodes(op_type)
    flags = [-1] * len(all_ops)
    res = []
    for idx, node in enumerate(all_ops):
        pre_node = oxgraph[node.inputs[0]]
        if pre_node in all_ops:
            next_idx = all_ops.index(pre_node)
            if flags[idx] == -1 and flags[next_idx] == -1:
                res.append([node, pre_node])
                flags[idx] =  flags[next_idx] = len(res) - 1
            elif flags[idx] != -1 and flags[next_idx] == -1:
                res[flags[idx]].append(pre_node)
                flags[next_idx] = flags[idx]
            elif flags[idx] == -1 and flags[next_idx] != -1:
                res_idx = res[flags[next_idx]].index(pre_node)
                res[flags[next_idx]].insert(res_idx, node)
                flags[idx] = flags[next_idx]
            else:
                res[flags[idx]] = res[flags[idx]] + res[flags[next_idx]]
                flags[next_idx] = flags[idx]
    flags = list(filter(lambda x: x != -1, flags))
    uniq_flags = []
    for f in flags:
        if f not in uniq_flags:
            uniq_flags.append(f)
    # 倒置
    return [res[idx][::-1] for idx in uniq_flags]


def fix_continuous_slice(graph):
    slice_node_list = get_continuous_op(graph, op_type='Slice')
    for nodes in slice_node_list:
        if len(nodes) > 2:
            raise NotImplementedError('Only support merge two nodes')
        node1, node2 = nodes
        merge_slice_node(graph, node1, node2)


def fix_onnx(graph):
    fix_continuous_slice(graph)
    convert_cast_nodes(graph)
    transfer_constantofshape(graph)
    convert_all_constants(graph)
    convert_all_initializers(graph)
    return graph

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    onnx_graph = OnnxGraph(input_path)
    onnx_graph = fix_onnx(onnx_graph)
    onnx_graph.save(output_path)
