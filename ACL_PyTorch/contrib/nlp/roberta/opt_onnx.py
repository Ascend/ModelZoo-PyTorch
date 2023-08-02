# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
 
 
import sys
from typing import Optional, List, Union
 
import numpy as np
from auto_optimizer import OnnxGraph, OnnxNode
 
 
def pattern_select(
    graph: OnnxGraph,
    candidate_nodes: Union[str, List[str]], 
    preorders: Optional[List[str]] = None, 
    successors: Optional[List[str]] = None
) -> List[OnnxNode]:
    ret = []
    preorders = preorders or []
    successors = successors or []
 
    if isinstance(candidate_nodes, str):
        candidate_nodes = graph.get_nodes(candidate_nodes)
    
    for node in candidate_nodes:
        pattern_check = True
        current_node = node
        for p in preorders[::-1]:
            if isinstance(p, str):
                op_type = p
                input_idx = 0
 
            elif isinstance(p, tuple):
                op_type, input_idx = p
 
            else:
                raise TypeError(f"Invalid preorder type: {type(p)}!")
 
            current_node = graph.get_prev_node(current_node.inputs[input_idx])
            if not current_node or current_node.op_type != op_type:
                pattern_check = False
                break
 
        if not pattern_check:
            continue
        
        current_node = node
        for s in successors:
            output_idx = 0
            if isinstance(s, str):
                op_type = s
 
            elif isinstance(s, tuple):
                op_type, output_idx = s
                
            else:
                raise TypeError(f"Invalid successor type: {type(s)}!")
 
            next_nodes = graph.get_next_nodes(current_node.outputs[output_idx])
            pattern_check = False
            for next_node in next_nodes:
                if next_node.op_type == op_type:
                    current_node = next_node
                    pattern_check = True
                    break
 
            if not pattern_check:
                break
 
        if pattern_check:
            ret.append(node)
    
    return ret
 
 
def insert_reshape_node(graph, anchor_node, dst_shape, mode='after'):
    inserted_reshape_node = graph.add_node(
        f"Reshape_{mode}_{anchor_node.name}",
        "Reshape",
    )
    inserted_reshape_init = graph.add_initializer(
        f"Reshape_init_{mode}_{anchor_node.name}",
        np.array(dst_shape, dtype="int64")
    )
    graph.insert_node(anchor_node.name, inserted_reshape_node, mode=mode)
    inserted_reshape_node.inputs.append(inserted_reshape_init.name)
 
 
def fix_cpu(graph):
    for cast_node in graph.get_nodes(op_type="Cast"):
        next_node = graph.get_next_nodes(cast_node.outputs[0])[0]
        if next_node.op_type == "Add":
            # int64 -> int32
            cast_node['to'] = 6
            inserted_add_init = graph.add_initializer(
                f"{next_node.name}_init",
                np.array(1, dtype='int32')
            )
            next_node.inputs[1] = inserted_add_init.name
 
 
def merge_axis(graph, seq, bs):
    # insert reshape node: 3 axis -> 2 axis
    # Pattern: Gather -> [Add]
    target_add = pattern_select(graph, 'Add', preorders=['Gather'])[0]
    insert_reshape_node(graph, target_add, [-1, 768])
 
    # Pattern: [Sub] -> Mul
    target_sub = pattern_select(graph, 'Sub', successors=['Mul'])[0]
    insert_reshape_node(graph, target_sub, [bs*seq, 1])
 
    # insert reshape node: 2 axis -> 3 axis
    # Pattern: [Gather] -> Gemm
    target_gather = pattern_select(graph, 'Gather', successors=['Gemm'])[0]
    insert_reshape_node(graph, target_gather, [-1, seq, 768], mode='before')
 
 
def opt_attention(graph, seq, bs):
    # remove first/last transpose node
    transpose_nodes = graph.get_nodes(op_type="Transpose")
    # sorted by op index
    transpose_nodes = sorted(transpose_nodes,
                             key=lambda node : int(node.name.split("_")[1]))
    graph.remove(transpose_nodes[0].name)
    graph.remove(transpose_nodes[-1].name)
 
    for softmax_node in graph.get_nodes(op_type="Softmax"):
        softmax_node['axis'] = -1
        # structure1:
        # reshape2(1/2)->transpose1(1/2)->matmul1->reshape1->where->reshape0->softmax
        reshape_node0 = graph.get_prev_node(softmax_node.inputs[0])
        where_node = graph.get_prev_node(reshape_node0.inputs[0])
        reshape_node1 = graph.get_prev_node(where_node.inputs[-1])
        matmul_node1 = graph.get_prev_node(reshape_node1.inputs[0])
        transpose_node1_1 = graph.get_prev_node(matmul_node1.inputs[0])
        reshape_node2_1 = graph.get_prev_node(transpose_node1_1.inputs[0])
        transpose_node1_2 = graph.get_prev_node(matmul_node1.inputs[1])
        reshape_node2_2 = graph.get_prev_node(transpose_node1_2.inputs[0])
 
        # opt reshape order && change transpose perm
        dst_shape_name = reshape_node1.inputs[1]
        graph[dst_shape_name].value = np.array([bs, seq, 12, 64], dtype="int64")
        transpose_node1_1['perm'] = [0, 2, 1, 3]
        transpose_node1_2['perm'] = [0, 2, 1, 3]
        reshape_node2_1.inputs[1] = dst_shape_name
        reshape_node2_2.inputs[1] = dst_shape_name
        graph.remove(reshape_node0.name)
        graph.remove(reshape_node1.name)
        # split transpose_node_1_2: [0, 2, 3, 1] -> [0, 2, 1, 3] + [0, 1, 3, 2]
        inserted_transpose_node = graph.add_node(
            f"Transpose_after_{transpose_node1_2.name}",
            "Transpose",
            attrs={
                "perm": [0, 1, 3, 2]
            }
        )
        graph.insert_node(transpose_node1_2.name, inserted_transpose_node)
        # unsqueeze->where ==> unsqueeze->cast->mul->expand->add
        # 1. insert add node
        unsqueeze_node = graph.get_prev_node(where_node.inputs[0])
        where_ori_input0 = where_node.inputs[0]
        inserted_add_node = graph.add_node(
            where_node.name.replace("Where", "Add"),
            "Add"
        )
        graph.insert_node(matmul_node1.name, inserted_add_node)
        inserted_add_node.inputs.append(where_ori_input0)
        graph.remove(where_node.name)
        softmax_node.inputs[0] = inserted_add_node.outputs[0]
        # 2. insert cast node
        inserted_cast_node = graph.add_node(
            f"Cast_after_{unsqueeze_node.name}",
            "Cast",
            attrs={
                'to': 1
            }
        )
        graph.insert_node(unsqueeze_node.name, inserted_cast_node)
        # 3. insert mul node
        inserted_mul_node = graph.add_node(
            f"Mul_after_{unsqueeze_node.name}",
            "Mul"
        )
        graph.insert_node(inserted_cast_node.name, inserted_mul_node)
        mul_init = graph.add_initializer(
            f"Mul_init_after_{unsqueeze_node.name}",
            np.array(-65504).astype("float32")
        )
        inserted_mul_node.inputs.append(mul_init.name)
        # 4. insert expand node
        inserted_expand_node = graph.add_node(
            f"Expand_after_{unsqueeze_node.name}",
            "Expand"
        )
        graph.insert_node(inserted_mul_node.name, inserted_expand_node)
        expand_init = graph.add_initializer(
            f"Expand_init_after_{unsqueeze_node.name}",
            np.array([bs, 1, seq, seq]).astype("int64")
        )
        inserted_expand_node.inputs.append(expand_init.name)
        # reconnect mul node: mul_node -> reshape_node2_1
        mul_node = graph.get_prev_node(reshape_node2_1.inputs[0])
        reshape_node2_1.inputs[0] = mul_node.inputs[0]
        mul_node.inputs[0] = matmul_node1.outputs[0]
        inserted_add_node.inputs[0] = mul_node.outputs[0]
        mul_node.name = "bert_" + mul_node.name
 
        # structure2:
        # softmax->matmul_node2<-transpose_node2->reshape_node3
        matmul_node2 = graph.get_next_nodes(softmax_node.outputs[0])[0]
        transpose_node2 = graph.get_prev_node(matmul_node2.inputs[1])
        reshape_node3 = graph.get_prev_node(transpose_node2.inputs[0])
        # change reshape/transpose paras
        transpose_node2['perm'] = [0, 2, 1, 3]
        reshape_node3.inputs[1] = dst_shape_name
 
        # structure3:
        # softmax->matmul_node2->transpose_node3->reshape_node4
        transpose_node3 = graph.get_next_nodes(matmul_node2.outputs[0])[0]
        reshape_node4 = graph.get_next_nodes(transpose_node3.outputs[0])[0]
        # change reshape/transpose paras
        transpose_node3 ['perm'] = [0, 2, 1, 3]
        graph[reshape_node4.inputs[1]].value = np.array([-1, 768], dtype='int64')
 
 
if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    bs = int(sys.argv[3])
    seq = int(sys.argv[4])
 
    onnx_graph = OnnxGraph.parse(input_path)
    fix_cpu(onnx_graph)
    merge_axis(onnx_graph, seq, bs)
    opt_attention(onnx_graph, seq, bs)
    onnx_graph.update_map()
    onnx_graph.remove_unused_nodes()
    onnx_graph.infershape()
    onnx_graph.save(save_path)