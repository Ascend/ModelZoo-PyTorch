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
from magiconnx.optimize.optimizer_manager import OptimizerManager


def merge_sub_ops(graph):
    def merge(sub_node1, sub_node2):
        # sub_node1->next_nodes ==> sub_node2->next_nodes
        for next_node in graph.get_next_nodes(sub_node1.name):
            for idx, input_name in enumerate(next_node.inputs):
                if input_name == sub_node1.outputs[0]:
                    next_node.inputs[idx] = sub_node2.outputs[0]

        # del sub_node1
        graph.del_node(sub_node1.name, auto_connection=False)

    for reducemean_node in graph.get_nodes(op_type="ReduceMean"):
        next_nodes = graph.get_next_nodes(reducemean_node.name)
        if len(next_nodes) == 2 and next_nodes[0].op_type == "Sub" and \
           next_nodes[1].op_type == "Sub":
            # check sub nodes with same inputs
            sub_node1 = next_nodes[0]
            sub_node2 = next_nodes[1]
            if sub_node1.inputs == sub_node2.inputs:
                merge(sub_node1, sub_node2)


def fix_mul(graph):
    # exchange constant value node as second input for mul op
    for mul_node in graph.get_nodes(op_type="Mul"):
        if len(mul_node.inputs) == 2:
            if graph[mul_node.inputs[0]].op_type == "Initializer":
                # exchange input nodes
                input_name1 = mul_node.inputs[0]
                mul_node.inputs[0] = mul_node.inputs[1]
                mul_node.inputs[1] = input_name1


def fix_mul_seq(graph, bs):
    # fix reshape value:
    # 1. 2dim: seq_len(dim=0)->-1
    # 2. 3dim: seq_len(dim=1)->-1, fix batchsize(dim=0)
    # 3. 4dim: seq_len(dim=1)->-1
    for reshape_node in graph.get_nodes(op_type="Reshape"):
        dst_shape = graph[reshape_node.inputs[1]].value.copy()
        if len(dst_shape) == 2 and dst_shape[0] != -1:
            dst_shape[0] = -1
        elif dst_shape[1] != -1 and len(dst_shape) > 2:
            dst_shape[1] = -1
            if len(dst_shape) == 3:
                dst_shape[0] = bs
        graph[reshape_node.inputs[1]].value = dst_shape

    # fix first add weight
    add_node = graph.get_nodes(op_type="Add")[0]
    add_value = graph[add_node.inputs[1]].value
    inserted_init = graph.add_initializer(
        "Inserted_init_rank",
        add_value
    )
    shape_node = graph.add_node(
        "Inserted_shape_rank",
        "Shape",
        inputs=[add_node.inputs[0]],
        outputs=["out_Inserted_shape_rank"]
    )
    gather_indices = graph.add_initializer(
        "Inserted_gather_indices_rank",
        np.array([1], dtype="int64")
    )
    gather_node = graph.add_node(
        "Inserted_gather_rank",
        "Gather",
        inputs=[shape_node.outputs[0], gather_indices.name],
        outputs=["out_Inserted_gather_rank"]
    )

    def build_slice_node(node_name, input_list):
        input_names = []
        for idx, input_value in enumerate(input_list):
            if isinstance(input_value, int):
                _init_node = graph.add_initializer(
                    f"input_{idx}_{node_name}",
                    np.array([input_value], dtype="int64")
                )
                input_names.append(_init_node.name)
            elif isinstance(input_value, str):
                input_names.append(input_value)
            else:
                raise NotImplementedError()
        return graph.add_node(
            node_name,
            "Slice",
            inputs=input_names,
            outputs=[f"out_{node_name}"]
        )

    slice_node = build_slice_node("Inserted_slice_rank", [inserted_init.name, 0, "out_Inserted_gather_rank", 1, 1])
    add_node.inputs[1] = slice_node.outputs[0]

    # fix expand value
    expand_node = graph.get_nodes(op_type="Expand")[0]
    expand_init = graph[expand_node.inputs[1]]
    expand_init.value = expand_init.value.copy()[:2]
    concat_node = graph.add_node(
        "Inserted_concat_rank",
        "Concat",
        attrs={
            'axis': 0
        },
        inputs=[expand_init.name, gather_node.outputs[0], gather_node.outputs[0]],
        outputs=['out_Inserted_concat_rank']
    )
    for expand_node in graph.get_nodes(op_type="Expand"):
        expand_node.inputs[1] = "out_Inserted_concat_rank"


if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    batch_size = None
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])
    onnx_graph = OnnxGraph(input_path)
    merge_sub_ops(onnx_graph)
    fix_mul(onnx_graph)
    optimize_manager_bert = OptimizerManager(onnx_graph, optimizers=["BertBigKernelOptimizer"])
    optimize_manager_bert.apply()
    if batch_size is not None:
        fix_mul_seq(onnx_graph, batch_size)
    onnx_graph.save(save_path)
