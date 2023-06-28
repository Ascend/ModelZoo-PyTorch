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


import sys
 
import numpy as np
from auto_optimizer import OnnxGraph
 
 
optimize_plans = {
     "vit_base_patch8_224": ["merge_bmm_axis", "pad_nz_block"],
    "vit_base_patch16_224": ["pad_nz_block"],
    "vit_base_patch16_384": ["merge_bmm_axis", "pad_nz_block"],
    "vit_base_patch32_224": ["merge_bmm_axis"],
    "vit_base_patch32_384": ["merge_bmm_axis", "pad_nz_block"],
}
 
 
def pattern_select(
    graph: OnnxGraph,
    candidate_nodes: list, 
    preorders: list = None, 
    successors: list = None
) -> list:
    ret = []
    preorders = preorders or []
    successors = successors or []
    
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
 
    
def get_attention_reshape_nodes(graph: OnnxGraph) -> list:
    # Pattern: Transpose -> [Reshape] -> MatMul
    all_reshape_nodes = graph.get_nodes("Reshape")
    return pattern_select(graph, all_reshape_nodes, ["Transpose"], ["MatMul"])
 
 
def get_layernorm_add_nodes(graph: OnnxGraph) -> list:
    # Pattern : Mul -> MatMul -> Add -> [Add]
    all_add_nodes = graph.get_nodes("Add")
    return pattern_select(graph, all_add_nodes, ["Mul", "MatMul", ("Add", 1)])
 
 
def get_layernorm_add_nodes_2(graph: OnnxGraph) -> list:
    # Pattern : Reshape -> MatMul -> Add -> [Add]
    all_add_nodes = graph.get_nodes("Add")
    return pattern_select(graph, all_add_nodes, ["Reshape", "MatMul", ("Add", 1)])
 
 
def merge_bmm_axis(graph: OnnxGraph, anchor_reshapes: list, anchor_adds: list) -> None:
    reshape_inits = list(set(node.inputs[1] for node in anchor_reshapes))
    original_shape = graph[reshape_inits[0]].value
    original_shape_init = graph.add_initializer(f"original_shape", original_shape)
 
    # change the target shape of reshape operators
    for _init in reshape_inits:
        b, x, y = graph[_init].value
        graph[_init].value = np.array([b * x, y])
 
    first_add_node = graph.get_nodes("Add")[0]
    next_add_node = [node for node in graph.get_next_nodes(first_add_node.outputs[0]) if node.op_type == "Add"][0]
    
    new_reshape_name = f"Reshape_before_{next_add_node.name}"
    graph.add_node(
        new_reshape_name, 
        "Reshape", 
        inputs=[first_add_node.outputs[0], reshape_inits[0]],
        outputs=[f"{new_reshape_name}/{next_add_node.name}"],
    )
    next_add_node.inputs[0] = f"{new_reshape_name}/{next_add_node.name}"
 
    # Restore the original shape temporarily for operator fusion
    for add_node in anchor_adds:
        output_name = add_node.outputs[0]
        new_reshape_name = f"Reshape_after_{add_node.name}"
        graph.add_node(
            new_reshape_name, 
            "Reshape", 
            inputs=[output_name, original_shape_init.name],
            outputs=[f"{new_reshape_name}_output"],
        )
 
        for next_node in graph.get_next_nodes(output_name):
            if next_node.op_type in ["ReduceMean", "Sub"]:
                next_node.inputs[next_node.inputs.index(output_name)] = f"{new_reshape_name}_output"
 
    # Restore the original shape at the end
    gather_node = graph.get_nodes("Gather")[0]
    new_reshape_name_2 = f"Reshape_before_{gather_node.name}"
    new_reshape_node_2 = graph.add_node(new_reshape_name_2, "Reshape")
    graph.insert_node(gather_node.name, new_reshape_node_2, mode="before")
    new_reshape_node_2.inputs.append(original_shape_init.name)
 
 
def cal_padding_shape(graph: OnnxGraph, merged: bool=False) -> tuple:
    first_reshape = graph.get_nodes("Reshape")[0]
    bs, hidden_dim1, hidden_dim2 = graph[first_reshape.inputs[1]].value
    hidden_dim2 += 1
 
    if hidden_dim2 % 16 == 0:
        padding_size = 0
    else:
        padding_size = int((hidden_dim2 // 16 + 1) * 16 - hidden_dim2)
 
    if merged:
        return (bs * padding_size, hidden_dim1), (bs * hidden_dim2, hidden_dim1)
 
    return (bs, padding_size, hidden_dim1), (bs, hidden_dim2, hidden_dim1)
 
 
def pad_nz_block(
        graph: OnnxGraph, 
        anchor_reshapes: list, 
        anchor_adds: list, 
        anchor_adds_2: list, 
        merged: bool=False
    ) -> None:
    padding_shape, original_shape = cal_padding_shape(graph, merged)
    axis = 0 if merged else 1
 
    new_concat_init = graph.add_initializer(f"padding_concat_init", np.zeros(padding_shape, dtype=np.float32))
    add_node = anchor_adds_2[0]
    new_concat_name = f"Concat_before_{add_node.name}"
    new_concat_node = graph.add_node(new_concat_name, "Concat", attrs={"axis": axis})
    graph.insert_node(add_node.name, new_concat_node, refer_index=0, mode="before")
    new_concat_node.inputs.append(new_concat_init.name)
    
 
    for reshape in anchor_reshapes:
        new_concat_name = f"Concat_after_{reshape.name}"
        new_concat_node = graph.add_node(new_concat_name, "Concat", attrs={"axis": axis})
        graph.insert_node(reshape.name, new_concat_node)
        new_concat_node.inputs.append(new_concat_init.name)
 
    for add_node in anchor_adds:
        output_name = add_node.outputs[0]
        new_slice_name = f"Slice_before_{add_node.name}"
        new_slice_init_starts = graph.add_initializer(f"{new_slice_name}_init_starts", np.array([0]))
        new_slice_init_ends = graph.add_initializer(f"{new_slice_name}_init_ends", np.array([original_shape[axis]]))
        new_slice_init_axes = graph.add_initializer(f"{new_slice_name}_init_axes", np.array([axis]))
        graph.add_node(
            new_slice_name, 
            "Slice",
            inputs=[output_name, new_slice_init_starts.name, new_slice_init_ends.name, new_slice_init_axes.name],
            outputs=[f"{new_slice_name}_output"],
            )
 
        for next_node in graph.get_next_nodes(output_name):
            if next_node.op_type in ["ReduceMean", "Sub", "Reshape"]:
                next_node.inputs[next_node.inputs.index(output_name)] = f"{new_slice_name}_output"
 
 
def apply_optimization(onnx_path: str, save_path:str, model_config: str) -> None:
    plan = optimize_plans.get(model_config)
    merged_axis = False
 
    g = OnnxGraph.parse(onnx_path)
    reshapes = get_attention_reshape_nodes(g)
    adds = get_layernorm_add_nodes(g)
    adds_2 = get_layernorm_add_nodes_2(g)
 
    for opt in plan:
        if opt == "merge_bmm_axis":
            merge_bmm_axis(g, reshapes, adds)
            merged_axis = True
 
        elif opt == "pad_nz_block":
            pad_nz_block(g, reshapes, adds, adds_2, merged_axis)
 
        g.update_map()
 
    g.remove_unused_nodes()
    g.infershape()
    g.save(save_path)
 
 
if __name__ == "__main__":
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    config = sys.argv[3]
 
    apply_optimization(input_model, output_model, config)