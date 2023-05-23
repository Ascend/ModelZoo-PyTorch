# Copyright 2021 Huawei Technologies Co., Ltd
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
from auto_optimizer import OnnxGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode, Node, Initializer


def get_config(graph):
    input_ph = graph.inputs[0]
    batch_size = input_ph.shape[0]
    seq_length = input_ph.shape[1]
    return batch_size, seq_length

def clip_outs(graph):
    outputs = graph.outputs[1:]
    for out in outputs:
        out_name = out.name
        concat_name = graph.get_prev_node(out_name).name
        left_input = graph.get_prev_node(graph[concat_name].inputs[0])
        right_input =graph.get_prev_node(graph[concat_name].inputs[1])
        left_input_unsqueeze_name = left_input.name
        right_input_Unsqueeze_name = right_input.name
        left_transpose_name = graph.get_prev_node(graph[left_input_unsqueeze_name].inputs[0]).name
        for name in [left_transpose_name,left_input_unsqueeze_name,right_input_Unsqueeze_name,concat_name,out_name]:
            graph.remove(name)
def clip_muls(graph):

    muls = graph.get_nodes(op_type='Mul')
    
    for mul in muls:
        
        prev_node = graph.get_prev_node(mul.inputs[0])
        next_node = graph.get_next_nodes(mul.outputs[0])[0]
        softmax_node = graph.get_next_nodes(next_node.outputs[0])[0]

        if prev_node.op_type == 'MatMul' and next_node.op_type == "Add" and softmax_node.op_type == "Softmax":
            graph.remove(mul.name)

def fix_ubfusion(graph):
    # reshape+gemm+reshape -> matmul+add
    for gemm_node in graph.get_nodes(op_type="Gemm"):
        pre_node = graph.get_prev_node(gemm_node.inputs[0])
        next_node = graph.get_next_nodes(gemm_node.outputs[0])[0]
        final_out = graph.get_next_nodes(next_node.outputs[0])

        if pre_node.op_type == "Reshape" and next_node.op_type == "Reshape":
            # del first reshape node
            graph.remove(pre_node.name)
            # gemm->bmm+add
            add_init = gemm_node.inputs[2]
            add_node = graph.add_node(
                f"Add_after_{gemm_node.name}",
                "Add"
            )
            graph.insert_node(gemm_node.name, add_node)
            add_node.inputs.append(add_init)
            matmul_node = graph.add_node(
                gemm_node.name.replace("Gemm", "MatMul"),
                "MatMul",
            )
            graph.insert_node(f"Add_after_{gemm_node.name}", matmul_node, mode='before')
            matmul_node.inputs.append(gemm_node.inputs[1])
            graph.remove(gemm_node.name)
            # del second reshape node
            graph.remove(next_node.name)


def fix_attention_qkv(graph, bs, seq_len):
    # fix reshape nodes
    # insert reshape before first add node
    reshape_before_add = graph.add_node(
        "reshape_before_add",
        "Reshape",
    )
    first_add = graph.get_nodes('Add')[0]
    add_init = graph.get_node(first_add.inputs[0], node_type=Initializer) or \
            graph.get_node(first_add.inputs[1], node_type=Initializer)
    if add_init.value.shape[0] == 1:
        add_init_value = np.tile(graph[add_init.name].value, (bs, 1, 1)).reshape([-1, 768])
    else:
        add_init_value = graph[add_init.name].value.reshape([-1, 768])
    graph[add_init.name].value = add_init_value

    graph.insert_node(first_add.name, reshape_before_add, mode='before')
    reshape_init = graph.add_initializer(
        f"{reshape_before_add.name}_value",
        np.array([-1, 768], dtype="int64")
    )
    reshape_before_add.inputs.append(reshape_init.name)

    # modify reshape in every subgraph
    for softmax in graph.get_nodes(op_type="Softmax"):
        matmul_node = graph.get_next_nodes(softmax.outputs[0])[0]
        transpose_node = graph.get_next_nodes(matmul_node.outputs[0])[0]
        reshape_node = graph.get_next_nodes(transpose_node.outputs[0])[0]
        reshape_int = graph[reshape_node.inputs[1]]
        reshape_int.value = np.array([-1, 768], dtype="int64")
    
    # fix branches of qkv
    for split_node in graph.get_nodes(op_type="Split"):
        add_node = graph.get_prev_node(split_node.inputs[0])
        matmul_node = graph.get_prev_node(add_node.inputs[0])
        start_node = graph.get_prev_node(matmul_node.inputs[0])

        # change transpose
        seen: List[List[int]] = []
        for idx in range(3):
            reshape_node = graph.get_next_nodes(split_node.outputs[idx])[0]
            transpose_node = graph.get_next_nodes(reshape_node.outputs[0])[0]
            perm: List[int] = transpose_node.attrs.get('perm', [1])
            if perm in seen:
                seen.remove(perm)
                query_perm = perm
            else:
                seen.append(perm)
                key_perm = perm
                key_transpose = transpose_node
             
        key_transpose.attrs["perm"] = query_perm
        new_perm = [query_perm.index(key_perm[i]) for i in range(len(key_perm))] # [0, 2, 3, 1] -> [0, 2, 1, 3] [0, 1, 3, 2]
        new_transpose = graph.add_node(
            name=f"{key_transpose.name}_after",
            op_type="Transpose",
            attrs={"perm": new_perm}
        )
        graph.insert_node(key_transpose.name, new_transpose, mode="after")
        
        # split matmul and add nodes
        weights_init = graph.get_node(matmul_node.inputs[0], node_type=Initializer) or \
            graph.get_node(matmul_node.inputs[1], node_type=Initializer)
        bias_init = graph.get_node(add_node.inputs[0], node_type=Initializer) or \
            graph.get_node(add_node.inputs[1], node_type=Initializer)
        weights = np.split(weights_init.value, 3, axis=-1)
        bias = np.split(bias_init.value, 3, axis=-1)
            
        for idx in range(3):
            new_matmul = graph.add_node(
                name=f"{matmul_node.name}_{idx}",
                op_type="MatMul",
                inputs=[start_node.outputs[0]],
                outputs=[f"{matmul_node.name}_{idx}_out"]
            )
            new_add = graph.add_node(
                name=f"{add_node.name}_{idx}",
                op_type="Add",
                inputs=[new_matmul.outputs[0]],
                outputs=[f"{add_node.name}_{idx}_out"]
            )
            new_matmul_value = graph.add_initializer(
                name=f"{new_matmul.name}_value",
                value=weights[idx]
            )
            new_add_value = graph.add_initializer(
                name=f"{new_add.name}_value",
                value=bias[idx]
            )
            new_matmul.inputs.append(new_matmul_value.name)
            new_add.inputs.append(new_add_value.name)
            reshape_node = graph.get_next_nodes(split_node.outputs[idx])[0]
            reshape_node.inputs[0] = new_add.outputs[0]
        graph.update_map()

        # insert add node for split
        another_add = graph.add_node(
            f"{start_node.name}_add",
            "Add",
            inputs=[start_node.outputs[0], first_add.outputs[0]]
        )

        # remove old nodes
        graph.remove(matmul_node.name)
        graph.remove(add_node.name)
        graph.remove(split_node.name)


def fix_attention_score(graph):
    # matmul+div+mul+sub -> mul+matmul++add 
    for softmax in graph.get_nodes(op_type="Softmax"):
        sub_node = graph.get_prev_node(softmax.inputs[0])
        mul_node = graph.get_prev_node(sub_node.inputs[0])
        div_node = graph.get_prev_node(mul_node.inputs[0])
        matmul_node = graph.get_prev_node(div_node.inputs[0])

        # div -> mul
        new_mul_node = graph.add_node(
            f"Mul_before_{matmul_node.name}",
            "Mul"
        )
        mul_init_value = np.array(1/graph[div_node.inputs[1]].value,dtype='float32')
        mul_init = graph.add_initializer(
            f"{new_mul_node.name}_value",
            mul_init_value
        )
        graph.insert_node(matmul_node.name, new_mul_node, mode="before")
        new_mul_node.inputs.append(f"{new_mul_node.name}_value")
        graph.remove(div_node.name)        

        # sub -> add
        add_node = graph.add_node(
            f"Add_before_{softmax.name}",
            "Add",
        )
        add_init_value = -graph[sub_node.inputs[1]].value
        add_init = graph.add_initializer(
            f"{add_node.name}_value",
            add_init_value
        )
        graph.insert_node(softmax.name, add_node, mode="before")
        add_node.inputs.append(f"{add_node.name}_value")
        graph.remove(sub_node.name)


if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
      
    onnx_graph = OnnxGraph.parse(input_path)
    graph_bs, graph_seq_len = get_config(onnx_graph)
    clip_outs(onnx_graph)
    fix_ubfusion(onnx_graph)
    fix_attention_qkv(onnx_graph, graph_bs, graph_seq_len)
    fix_attention_score(onnx_graph)
    clip_muls(onnx_graph)
    onnx_graph.save(save_path)
