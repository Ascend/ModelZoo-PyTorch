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
 
import os
import argparse
 
import numpy as np
from auto_optimizer import OnnxGraph, OnnxNode
from auto_optimizer.pattern.knowledges.big_kernel.knowledge_big_kernel import KnowledgeBigKernel
from modelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig
 
 
def pattern_select(
    graph: OnnxGraph,
    candidate_nodes: list, 
    preorders: list = None, 
    successors: list = None
) -> list:
    # Utile function: Select nodes by matching pattern
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
                raise TypeError(f'Invalid preorder type: {type(p)}!')
 
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
                raise TypeError(f'Invalid successor type: {type(s)}!')

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
 
 
def fix_mul(graph: OnnxGraph) -> None:
    # exchange constant value node as second input for mul op
    initializers = [init.name for init in graph.initializers]
    for mul_node in graph.get_nodes("Mul"):
        if len(mul_node.inputs) == 2:
            if mul_node.inputs[0] in initializers:
                # exchange input nodes
                mul_node.inputs = mul_node.inputs[::-1]
 
 
def fix_big_kernel(graph: OnnxGraph) -> None:
    all_add = graph.get_nodes('Add')
 
    # Pattern: [Add] -> MatMul -> Add -> Reshape
    bk_start = pattern_select(graph, all_add, [], ['MatMul', 'Add', 'Reshape'])[0]
 
    # Pattern: Reshape -> MatMul -> Add -> [Add]
    bk_end = pattern_select(graph, all_add, ['Reshape', ('MatMul', 1), ('Add', 1)])[0]
 
    knowledge = KnowledgeBigKernel(graph, bk_start.name, bk_end.name)
    match_results = knowledge.match_pattern(graph)
    for match_result in match_results:
        knowledge.apply(graph, match_result)
        
    knowledge.post_process(graph)
    fix_mul(graph)
 
 
def fix_mul_seq(graph: OnnxGraph, bs: int) -> None:
    # fix reshape value:
    # 1. 2dim: seq_len(dim=0)->-1
    # 2. >=3dim: seq_len(dim=1)->-1, batchsize(dim=0)->bs
    for reshape_node in graph.get_nodes(op_type="Reshape"):
        dst_shape = graph[reshape_node.inputs[1]].value.copy()
        if len(dst_shape) == 2 and dst_shape[0] != -1:
            dst_shape[0] = -1
        elif dst_shape[1] != -1 and len(dst_shape) > 2:
            dst_shape[1] = -1
            if len(dst_shape) >= 3:
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

    def build_slice_node(node_name: str, input_list: list) -> OnnxNode:
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


def get_quant_block_list(graph: OnnxGraph) -> list:
    # Do not apply the quantilization on QKV MatMuls.
 
    # Select MatMuls by matching the pattern: [MatMul] -> Add -> Reshape
    all_matmuls = graph.get_nodes('MatMul')
    qkv_matmuls = pattern_select(graph, all_matmuls, [], ['Add', 'Reshape'])
 
    # Return a list of names
    return [node.name for node in qkv_matmuls]
 
 
def quantilize_model(
    origial_onnx: str, 
    save_path: str, 
    block_nodes: list = None,
    optimize_graph: bool = True,
    disable_first_layer: bool = True,
    disable_last_layer: bool = True,
    sigma: int = 25,
) -> None:
 
    config = QuantConfig()
    config.sigma = sigma
    config.is_optimize_graph = optimize_graph
    config.disable_names = block_nodes if block_nodes else []
    config.disable_first_layer = disable_first_layer
    config.disable_last_layer = disable_last_layer
 
    calib = OnnxCalibrator(origial_onnx, config)
    calib.run()
    calib.export_quant_onnx(save_path)
 
 
def save_onnx(graph: OnnxGraph, save_path: str):
    graph.update_map()
    graph.remove_unused_nodes()
    graph.infer_shape()
    graph.save(save_path)
 
 
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_onnx_path', type=str, help='Path of the original ONNX.')
    parser.add_argument('save_onnx_path', type=str, help='Path to save the result ONNX.')
    parser.add_argument('-bk', '--fix_big_kernel', action='store_true',
                        help='Fix model to adapt to the big kernel optimization.')
    parser.add_argument('-q', '--quantilize', action='store_true', help='Apply the quantilze optimization.')
    parser.add_argument('-r', '--rank', action='store_true', help='Apply the multi-rank optimization.')
    return parser.parse_args()
 
 
if __name__ == '__main__':
    args = parse_arguments()
    input_path = args.input_onnx_path
    save_onnx_path = args.save_onnx_path
    onnx_graph = OnnxGraph.parse(input_path)
 
    if not (args.fix_big_kernel or args.quantilize or args.rank):
        print('Nothing to do.')
 
    else:
        if args.fix_big_kernel:
            fix_big_kernel(onnx_graph)
 
        if args.quantilize:
            file_name, ext = os.path.splitext(input_path)
            temp_path = file_name + '_temp' + ext
            save_onnx(onnx_graph, temp_path)
            
            if os.path.exists(save_onnx_path):
                os.remove(save_onnx_path)
 
            block_nodes_list = get_quant_block_list(onnx_graph)
            quantilize_model(
                temp_path, 
                save_onnx_path, 
                block_nodes_list, 
                optimize_graph=False, 
                disable_first_layer=False, 
                sigma=0,
            )
            onnx_graph = OnnxGraph.parse(save_onnx_path)
            os.remove(temp_path)
 
        if args.rank:
            batch_size = onnx_graph.inputs[0].shape[0]
            fix_mul_seq(onnx_graph, batch_size)
        
        save_onnx(onnx_graph, save_onnx_path)
        print('Done.')
