# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#


import sys
import numpy as np
from copy import deepcopy
from magiconnx import OnnxGraph


def fix_mul(graph, max_value=65504):
    for mul_node in graph.get_nodes(op_type="Mul"):
        for input_name in mul_node.inputs:
            input_node = graph[input_name]
            if input_node.op_type == "Constant" or input_node.op_type == "Initializer":
                fixed_value = deepcopy(input_node.value)
                value_mask_pos = (fixed_value > 65504)
                fixed_value[value_mask_pos] = 65504
                value_mask_neg = (fixed_value < -65504)
                fixed_value[value_mask_neg] = -65504
                if np.sum(value_mask_pos) > 0 or np.sum(value_mask_neg) > 0:
                    print(f"Fix value node: {input_name}")
                    input_node.value = fixed_value


def transfer_gather(graph, gather_node):
    # gather -> slice+unsqueeze
    node_num = int(gather_node.name.split("_")[-1])
    indices_value = graph[gather_node.inputs[1]].value
    axis = gather_node['axis']
    if indices_value > 0:
        raise ValueError("Gather to be fixed must has negativate indices")

    slice_node = graph.add_node(
        f"Slice_{node_num}",
        "Slice"
    )
    slice_start = graph.add_initializer(
        f"Slice_{node_num}_start",
        np.array([indices_value], dtype="int64")
    )
    end_value = indices_value + 1
    if indices_value == -1:
        end_value = 65504
    slice_end = graph.add_initializer(
        f"Slice_{node_num}_end",
        np.array([end_value], dtype="int64")
    )
    slice_axes = graph.add_initializer(
        f"Slice_{node_num}_axes",
        np.array([axis], dtype="int64")
    )
    squeeze_node = graph.add_node(
        f"Squeeze_{node_num}",
        "Squeeze",
        attrs={
            "axes": [axis]
        }
    )

    graph.insert_node(gather_node.name, slice_node, mode='after')
    slice_node.inputs.append(f"Slice_{node_num}_start")
    slice_node.inputs.append(f"Slice_{node_num}_end")
    slice_node.inputs.append(f"Slice_{node_num}_axes")
    graph.insert_node(slice_node.name, squeeze_node, mode='after')
    graph.del_node(gather_node.name)


def fix_gather(graph):
    for gather_node in graph.get_nodes(op_type="Gather"):
        gather_indices = graph[gather_node.inputs[1]]
        if gather_indices.op_type in ["Initializer", "Constant"] and \
           gather_indices.value == -1:
            transfer_gather(graph, gather_node)
            print(f"Fix {gather_node.name} succeed.")


if __name__ == '__main__':
    onnx_graph = OnnxGraph(sys.argv[1])
    fix_mul(onnx_graph)
    fix_gather(onnx_graph)
    onnx_graph.save(sys.argv[2])
