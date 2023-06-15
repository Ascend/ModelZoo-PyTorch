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

import argparse
from typing import List

import numpy as np
from auto_optimizer import OnnxGraph
from auto_optimizer.graph_refactor.interface.base_node import Initializer


def parse_args():
    parser = argparse.ArgumentParser(description="fix albert onnx")
    parser.add_argument("--input_file", type=str, required=True,
                        help="path to pth model")
    parser.add_argument("--output_file", type=str, required=True,
                        help="path to save onnx model")
    parser.add_argument("--model_size", type=str, default='base',
                        help="model_size of bert", choices=['base', 'large'])
    args_out = parser.parse_args()
    return args_out


def get_config(graph):
    input_ph = graph.inputs[0]
    bs, seq_len = input_ph.shape[0], input_ph.shape[1]
    return bs, seq_len


def fix_attention_lnqkv(graph, qkv_start_node):
    # insert reshape before qkv_start_node
    reshape_before_add = graph.add_node(
        f"Reshape_before_{qkv_start_node.name}",
        "Reshape"
    )
    reshape_init = graph.add_initializer(
        f"{reshape_before_add.name}_value",
        np.array([-1, HIDDEN_NUM], dtype="int64")
    )
    if graph.get_node(qkv_start_node.inputs[0], node_type=Initializer):
        graph.insert_node(qkv_start_node.name, reshape_before_add, refer_index=1, mode="before")
    else:
        graph.insert_node(qkv_start_node.name, reshape_before_add, refer_index=0, mode="before")
    reshape_before_add.inputs.append(reshape_init.name)

    # change transpose node
    seen: List[List[int]] = []
    next_nodes = graph.get_next_nodes(qkv_start_node.outputs[0])
    matmul_nodes = [n for n in next_nodes if n.op_type == "MatMul"]
    for idx in range(3):
        matmul_node = matmul_nodes[idx]
        add_node = graph.get_next_nodes(matmul_node.outputs[0])[0]
        reshape_node = graph.get_next_nodes(add_node.outputs[0])[0]
        transpose_node = graph.get_next_nodes(reshape_node.outputs[0])[0]
        perm: List[int] = transpose_node.attrs.get('perm', [1])
        if perm in seen:
            seen.remove(perm)
            query_perm = perm
        else:
            seen.append(perm)
            key_perm = perm
            key_transpose = transpose_node
        
    # [0, 2, 3, 1] -> [0, 2, 1, 3] [0, 1, 3, 2]
    key_transpose.attrs["perm"] = query_perm
    new_perm = [query_perm.index(key_perm[i]) for i in range(len(key_perm))]
    new_transpose = graph.add_node(
        name=f"{key_transpose.name}_after",
        op_type="Transpose",
        attrs={"perm": new_perm}
    )
    graph.insert_node(key_transpose.name, new_transpose, mode="after")


def fix_attention_score(graph, softmax_node, bs, seq_len):
    # fix reshape node 
    matmul_node = graph.get_next_nodes(softmax_node.outputs[0])[0]
    transpose_node = graph.get_next_nodes(matmul_node.outputs[0])[0]
    reshape_node = graph.get_next_nodes(transpose_node.outputs[0])[0]
    reshape_init = graph[reshape_node.inputs[1]]
    reshape_init.value = np.array([-1, HIDDEN_NUM], dtype="int64")

    # replace div node with mul node
    # insert expand node
    add_node = graph.get_prev_node(softmax_node.inputs[0])
    prev_node = graph.get_prev_node(add_node.inputs[0])
    if prev_node.op_type == "Div":
        div_node = prev_node
        refer_index = 0
    else:
        div_node = graph.get_prev_node(add_node.inputs[1])
        refer_index = 1
    
    div_init = graph.get_node(div_node.inputs[0], node_type=Initializer) or \
            graph.get_node(div_node.inputs[1], node_type=Initializer)
    mul_node = graph.add_node(
        f"bert_Mul_before_{add_node.name}",
        "Mul",
    )
    mul_init_value = np.array(1/div_init.value, dtype="float32")
    mul_init = graph.add_initializer(
        f"{mul_node.name}_value",
        mul_init_value
    )
    graph.insert_node(add_node.name, mul_node, refer_index=refer_index, mode="before")
    mul_node.inputs.append(mul_init.name)
    graph.remove(div_node.name)

    expand_node = graph.add_node(
        f"Expand_before_{add_node.name}",
        "Expand"
    )
    expand_init = graph.add_initializer(
        f"{expand_node.name}_value",
        np.array([bs, 1, seq_len, seq_len], dtype="int64")
    )
    graph.insert_node(add_node.name, expand_node, refer_index=~refer_index, mode="before")
    expand_node.inputs.append(expand_init.name)
    

def main(graph):
    # get config
    bs, seq_len = get_config(graph)
    
    # fix_lnqkv
    add_nodes = graph.get_nodes("Add")
    for add_node in add_nodes:
        if len(graph.get_next_nodes(add_node.outputs[0])) == 4:
            fix_attention_lnqkv(graph, add_node)
    
    # fix_attentionscore
    softmax_nodes = graph.get_nodes("Softmax")
    for softmax_node in softmax_nodes:
        fix_attention_score(graph, softmax_node, bs, seq_len)

    # insert last reshape to recover shape
    last_add = graph.get_nodes(op_type="Add")[-2]
    last_reshape = graph.add_node(
        "last_reshape",
        "Reshape"
    )
    reshape_init = graph.add_initializer(
        f"{last_reshape.name}_value",
        np.array([bs, seq_len, HIDDEN_NUM], dtype="int64")
    )
    if graph.get_node(last_add.inputs[0], node_type=Initializer):
        graph.insert_node(last_add.name, last_reshape, refer_index=1, mode="before")
    else:
        graph.insert_node(last_add.name, last_reshape,  refer_index=0, mode="before")
    last_reshape.inputs.append(reshape_init.name)

                                                     
if __name__=="__main__":
    args = parse_args()
    if args.model_size == "base":
        HIDDEN_NUM=768
    else:
        HIDDEN_NUM=1024

    onnx_graph = OnnxGraph.parse(args.input_file)
    main(onnx_graph)
    onnx_graph.infershape()
    onnx_graph.save(args.output_file)
