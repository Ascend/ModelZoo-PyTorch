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
import numpy as np
from auto_optimizer import OnnxGraph, OnnxPlaceHolder
from graph_fusion import keep_dynamic_batch


def generate_seq_mask(graph, input_node):
    def bfs_search(cur_node):
        cur_nodes = [cur_node]
        node_set = set()
        while cur_nodes:
            next_nodes = []
            for cur_node in cur_nodes:
                node_set.add(cur_node.name)
                for out_name in cur_node.outputs:
                    for next_node in graph.get_next_nodes(out_name):
                        next_nodes.append(next_node)
            if cur_nodes[0].op_type == "Slice" and next_nodes[0].op_type == "Slice":
                node_set.remove(cur_node.name)
                break
            if cur_nodes[0] in graph.outputs or next_nodes[0] in graph.outputs:
                break
            cur_nodes = next_nodes
        return node_set, cur_nodes[0], next_nodes[0]

    # get mask block input node
    pre_node_set, match_slice_node, end_slice_node = bfs_search(
        graph.get_next_nodes(input_node.name)[0])
    # clean sub/mul node && reconnect next node
    next_node = graph.get_next_nodes(end_slice_node.outputs[0])[0]
    graph.remove(match_slice_node.name)
    graph.remove(end_slice_node.name)
    mask_input = OnnxPlaceHolder('mask', dtype=np.dtype("float32"), shape=['batch', 'seq'])
    graph.inputs.append(mask_input)
    next_node.inputs[0] = mask_input.name
    return graph


def generate_postion_mask(graph, node_list, batch, feature_size=4):
    # build position mask input node
    mask_input = OnnxPlaceHolder('position', dtype=np.dtype("float32"), shape=['batch', 'hidden_size'])
    graph.inputs.append(mask_input)

    cast_node = graph.add_node(
        "Cast_first",
        "Cast",
        attrs={
            'to': 6
        },
        inputs=['position'],
        outputs=['out_Cast_first']
    )

    def build_split_node(pre_node, input_name, axis, split_list):
        split_node = graph.add_node(
            f"Split_after_{pre_node.name}",
            "Split",
            attrs={
                'axis': axis,
                'split': split_list
            },
            inputs=[input_name],
            outputs=[f"out_{idx}_Split_after_{pre_node.name}" for idx in range(len(split_list))]
        )
        return split_node
    input_split_node = build_split_node(cast_node, cast_node.outputs[0], 0, [1] * batch)

    def build_mul_node(input_list, node_name):
        mul_node = graph.add_node(
            node_name,
            "Mul",
            inputs=input_list,
            outputs=[f"out_{node_name}"]
        )
        return mul_node

    # replace slice: split->multi gather->concat
    for node_name in node_list:
        node = graph[node_name]
        pre_node = graph.get_prev_node(node.inputs[0])
        assert pre_node.op_type == "Reshape"
        # merge dst shape
        concat_node = graph.get_prev_node(pre_node.inputs[1])
        mul_node1 = build_mul_node(concat_node.inputs[:2], f"Mul_before_{concat_node.name}_1")
        mul_node2 = build_mul_node(concat_node.inputs[2:], f"Mul_before_{concat_node.name}_2")
        concat_node.inputs = [mul_node1.outputs[0], mul_node2.outputs[0]]

        split_node = build_split_node(pre_node, pre_node.outputs[0], 0, [feature_size] * batch)
        split_out_list = []
        for _idx, split_out_name in enumerate(split_node.outputs):
            _name = node.name.replace("Slice", "Gather") + f"_{str(_idx)}"
            gather_node = graph.add_node(
                _name,
                "Gather",
                attrs={
                    'axis': 1
                },
                inputs=[split_out_name, input_split_node.outputs[_idx]],
                outputs=[f"out_{_name}"]
            )
            split_out_list.append(f"out_{_name}")
        concat_node = graph.add_node(
            f"Concat_after_{pre_node.name}",
            "Concat",
            attrs={'axis': 0},
            inputs=split_out_list,
            outputs=[f"out_Concat_after_{pre_node.name}"]
        )
        next_nodes = graph.get_next_nodes(node.outputs[0])
        graph.remove(node_name, mapping={})
        for next_node in next_nodes:
            for _i, _input_name in enumerate(next_node.inputs):
                if _input_name == node.outputs[0]:
                    next_node.inputs[_i] = f"out_Concat_after_{pre_node.name}"
    return graph

def generate_conv_mask(graph, node_list):
    mask_input = OnnxPlaceHolder("conv_mask", dtype=np.dtype("float32"), shape=['batch', 1, 'seq'])
    graph.inputs.append(mask_input)

    for node_name in node_list:
        node = graph[node_name]
        mul_node = graph.add_node(
            f"Mul_before_{node.name}",
            "Mul"
        )
        graph.insert_node(node_name, mul_node, mode='before')
        mul_node.inputs.append('conv_mask')
    return graph


def get_position_mask_inputs(graph):
    slice_node_list = []
    for node in graph.get_nodes(op_type="Slice"):
        pre_node = onnx_graph.get_prev_node(node.inputs[0])
        if pre_node is not None and pre_node.op_type == "Reshape":
            pre_node = onnx_graph.get_prev_node(pre_node.inputs[0])
            if pre_node is not None and pre_node.op_type == "Concat":
                slice_node_list.append(node.name)
    return slice_node_list


def get_conv_mask_inputs(grapg):
    conv_node_list = []
    for node in onnx_graph.get_nodes(op_type="Conv"):
        weight = onnx_graph[node.inputs[1]]
        if weight.value.shape[-1] == 15:
            conv_node_list.append(node.name)
    return conv_node_list


if __name__ == '__main__':
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    batch = int(sys.argv[3])
    onnx_graph = OnnxGraph.parse(input_model)
    keep_dynamic_batch(onnx_graph)
    onnx_graph = generate_seq_mask(onnx_graph, onnx_graph.inputs[0])
    pos_input_list = get_position_mask_inputs(onnx_graph)
    onnx_graph = generate_postion_mask(onnx_graph, pos_input_list, batch)
    conv_input_list = get_conv_mask_inputs(onnx_graph)
    onnx_graph = generate_conv_mask(onnx_graph, conv_input_list)
    onnx_graph.remove_unused_nodes()
    del onnx_graph.outputs[1]
    onnx_graph.save(output_model)
