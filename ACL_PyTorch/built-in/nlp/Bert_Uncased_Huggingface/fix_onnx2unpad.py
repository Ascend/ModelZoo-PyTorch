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
import numpy as np
from auto_optimizer import OnnxGraph
from auto_optimizer.graph_refactor.interface.base_node import Initializer

def parse_args():
    parser = argparse.ArgumentParser(description="fix bert onnx")
    parser.add_argument("--input_file", type=str, required=True,
                        help="path to pth model")
    parser.add_argument("--output_file", type=str, required=True,
                        help="path to save onnx model")
    arg = parser.parse_args()
    return arg

def modify_inputs(graph):
    input_ids = graph["input_ids"]
    nodes_after_input_ids = graph.get_next_nodes(input_ids.name)
    for node in nodes_after_input_ids:
        if node.op_type == "Shape":
            shape_node = node
            break
    gather_node = graph.get_next_nodes(shape_node.outputs[0])[0]
    cast = graph.add_node("cast_after_gather", "Cast", attrs={"to": 6})
    graph.insert_node(gather_node.name, cast, refer_index=0, mode="after")
    range_node = graph.add_node("range_for_position", "Range")
    graph.insert_node(cast.name, range_node, refer_index=0, mode="after")
    start = graph.add_initializer("start", value=np.array(0, dtype="int32"))
    delta = graph.add_initializer("delta", value=np.array(1, dtype="int32"))
    range_node.inputs.extend([start.name, delta.name])
    range_node.inputs[0], range_node.inputs[1] = range_node.inputs[1], range_node.inputs[0]
    unsqueeze_node = graph.add_node("unsqueeze_after_range", "Unsqueeze",
                                    inputs=[range_node.outputs[0]], outputs=["unsqueeze_after_range_out"],
                                    attrs={"axes": np.array([0], dtype=np.int64)})
    next_node = graph.get_next_nodes(range_node.outputs[0])[0]
    while next_node.op_type != "Gather":
        next_node = graph.get_next_nodes(next_node.outputs[0])[0]
    gather_position = next_node
    shape2 = graph.add_node("shape_of_inputs", "Shape",
                            inputs=[input_ids.name], outputs=["shape_of_inputs_out"])
    expand_node = graph.add_node("expand_position_ids", "Expand",
                                 inputs=[unsqueeze_node.outputs[0], shape2.outputs[0]], outputs=["expand_out"])
    gather_position.inputs[1] = expand_node.outputs[0]

def remove_after_attention_mask_nodes(graph):
    unsqueezes = graph.get_nodes("Unsqueeze")
    for unsqueeze in unsqueezes:
        if unsqueeze.inputs[0] == "attention_mask":
            unsqueeze2 = graph.get_next_nodes(unsqueeze.outputs[0])[0]
            cast = graph.get_next_nodes(unsqueeze2.outputs[0])[0]
            sub = graph.get_next_nodes(cast.outputs[0])[0]
            mul = graph.get_next_nodes(sub.outputs[0])[0]
            graph.remove(unsqueeze.name, {})
            graph.remove(unsqueeze2.name, {})
            graph.remove(cast.name, {})
            graph.remove(sub.name, {})
            graph.remove(mul.name, {})
            break

def add_genseqlen(graph):
    attention_mask = graph["attention_mask"]
    cast_after_attention_mask = graph.add_node("cast_after_attention_mask", "Cast",
                                               inputs=[attention_mask.name], outputs=[f"{attention_mask.name}.1"],
                                               attrs={"to": 6})
    genseqlen = graph.add_node("genseqlen", "GenSeqLen",
                               inputs=[cast_after_attention_mask.outputs[0]], outputs=["seqlen", "seqlen_ori"])

def add_ntokens(graph):
    genseqlen = graph["genseqlen"]
    cast_node1 = graph.add_node("cast_after_genseqlen", "Cast",
                                inputs=[genseqlen.outputs[0]], outputs=[f"{genseqlen.name}.1"],
                                attrs={"to": 7})
    reducesum_node = graph.add_node(f"reducesum_after_{cast_node1.name}", "ReduceSum",
                                    inputs=[cast_node1.outputs[0]], outputs=["ntokens_before"],
                                    attrs={"keepdims": 1, "axes": np.array([-1], dtype=np.int64)})
    cast_node2 = graph.add_node("ntokens", "Cast",
                                inputs=[reducesum_node.outputs[0]], outputs=["ntokens_out"],
                                attrs={"to": 6})

def add_mask(graph):
    mask = np.zeros((16, 16), dtype=np.float16)
    for i in range(16):
        for j in range(15 - i, 16):
            mask[i][j] = -10000.0
    mask_ini = graph.add_initializer("spMask", value=mask)

def add_unpadinput(graph):
    genseqlen = graph["genseqlen"]
    add_nodes = graph.get_nodes("Add")
    ntokens = graph["ntokens"]
    for add_node in add_nodes:
        if len(graph.get_next_nodes(add_node.outputs[0])) == 4:
            cast = graph.add_node("cast_after_input_embedding", "Cast", attrs={"to": 10})
            graph.insert_node(add_node.name, cast, refer_index=0, mode="after")
            unpadinput = graph.add_node("unpadinput", "UnpadInput")
            graph.insert_node(cast.name, unpadinput, refer_index=0, mode="after")
            unpadinput.inputs.extend([genseqlen.outputs[0], ntokens.outputs[0]])
            cast = graph.add_node("cast_after_unpadinput", "Cast",
                                  inputs=[unpadinput.outputs[0]], outputs=[f"{unpadinput.name}.1"],
                                  attrs={"to": 1})
            graph.insert_node(unpadinput.name, cast, refer_index=0, mode="after")
            break

def add_gather_bs_max_hidden(graph):
    unpadinput_node = graph["unpadinput"]
    cast_node = graph.get_prev_node(unpadinput_node.inputs[0])
    add_node = graph.get_prev_node(cast_node.inputs[0])
    shape_node1 = graph.add_node("shape_of_add1", "Shape",
                                 inputs=[add_node.outputs[0]], outputs=["shape_of_add_out1"])
    shape_node2 = graph.add_node("shape_of_add2", "Shape",
                                 inputs=[add_node.outputs[0]], outputs=["shape_of_add_out2"])
    shape_node3 = graph.add_node("shape_of_add3", "Shape",
                                 inputs=[add_node.outputs[0]], outputs=["shape_of_add_out3"])
    indices_batch = graph.add_initializer("indices_of_batch", value=np.array(0, dtype=np.int64))
    indices_maxseqlen = graph.add_initializer("indices_of_maxseqlen", value=np.array(1, dtype=np.int64))
    indices_hidden = graph.add_initializer("indices_of_hidden", value=np.array(2, dtype=np.int64))
    gather_batch = graph.add_node("batch", "Gather",
                                  inputs=[shape_node1.outputs[0]], outputs=["model_batch"])
    gather_maxseqlen = graph.add_node("maxseqlen", "Gather",
                                      inputs=[shape_node2.outputs[0]], outputs=["model_seqlen"])
    gather_hidden = graph.add_node("hidden", "Gather",
                                   inputs=[shape_node3.outputs[0]], outputs=["model_hidden"])
    gather_batch.inputs.append(indices_batch.name)
    gather_maxseqlen.inputs.append(indices_maxseqlen.name)
    gather_hidden.inputs.append(indices_hidden.name)

def modify_self_attention(graph, input_node, layer):
    genseqlen = graph["genseqlen"]
    nodes4 = graph.get_next_nodes(input_node.outputs[0])

    weight_bias = {}
    matmul_name = []  # 记录matmul节点的name，后续需要删掉
    for node in nodes4:
        if node.op_type == "Add":
            addln_node = node
            dense_bias_node = graph.get_prev_node(addln_node.inputs[0])
        if node.op_type == "MatMul":
            matmul_name.append(node.name)
            bias = graph.get_next_nodes(node.outputs[0])[0]
            for i in range(len(bias.inputs)):
                if "query" in bias.inputs[i]:
                    weight_bias["query"] = [node.inputs[1], bias.inputs[i]]
                elif "key" in bias.inputs[i]:
                    weight_bias["key"] = [node.inputs[1], bias.inputs[i]]
                elif "value" in bias.inputs[i]:
                    weight_bias["value"] = [node.inputs[1], bias.inputs[i]]
    for name in matmul_name:
        graph.remove(name, {})

    # 将qkv的matmul修改为Gemm
    gemm_query = graph.add_node(f"gemm_layer{layer}_query", "Gemm",
                                inputs=[input_node.outputs[0]], outputs=[f"gemm_layer{layer}_query_out"],
                                attrs={"alpha": 1, "beta": 1, "transB": 1})
    gemm_key = graph.add_node(f"gemm_layer{layer}_key", "Gemm",
                                inputs=[input_node.outputs[0]], outputs=[f"gemm_layer{layer}_key_out"],
                                attrs={"alpha": 1, "beta": 1, "transB": 1})
    gemm_value = graph.add_node(f"gemm_layer{layer}_value", "Gemm",
                                inputs=[input_node.outputs[0]], outputs=[f"gemm_layer{layer}_value_out"],
                                attrs={"alpha": 1, "beta": 1, "transB": 1})
    query_weight = graph.get_node(weight_bias["query"][0], Initializer)
    key_weight = graph.get_node(weight_bias["key"][0], Initializer)
    value_weight = graph.get_node(weight_bias["value"][0], Initializer)
    query_bias = graph.get_node(weight_bias["query"][1], Initializer)
    key_bias = graph.get_node(weight_bias["key"][1], Initializer)
    value_bias = graph.get_node(weight_bias["value"][1], Initializer)
    query_weight.value = query_weight.value.T
    key_weight.value = key_weight.value.T
    value_weight.value = value_weight.value.T
    
    gemm_query.inputs.extend([query_weight.name, query_bias.name])
    gemm_key.inputs.extend([key_weight.name, key_bias.name])
    gemm_value.inputs.extend([value_weight.name, value_bias.name])

    cast_query = graph.add_node(f"cast_after_gemm_query_layer{layer}", "Cast",
                                attrs={"to": 10})
    cast_key = graph.add_node(f"cast_after_gemm_key_layer{layer}", "Cast",
                                attrs={"to": 10})
    cast_value = graph.add_node(f"cast_after_gemm_value_layer{layer}", "Cast",
                                attrs={"to": 10})
    graph.insert_node(gemm_query.name, cast_query, refer_index=0, mode="after")
    graph.insert_node(gemm_key.name, cast_key, refer_index=0, mode="after")
    graph.insert_node(gemm_value.name, cast_value, refer_index=0, mode="after")

    # 添加flashattention节点
    flashattention = graph.add_node(f"flashattention_layer{layer}", "FlashAttention",
                                    inputs=[cast_query.outputs[0], cast_key.outputs[0], cast_value.outputs[0]],
                                    outputs=[f"flashattention_layer{layer}_out"])
    gather_batch = graph["batch"]
    flashattention.inputs.extend([genseqlen.outputs[1], gather_batch.outputs[0], "spMask"])

    # flashattention后的cast
    cast_after_flashattention = graph.add_node(f"cast_after_layer{layer}_flashattention", "Cast",
                                               inputs=[flashattention.outputs[0]],
                                               outputs=[f"{flashattention.name}.1"],
                                               attrs={"to": 1})
    
    # dense
    for i in range(len(dense_bias_node.inputs)):
        if graph.get_prev_node(dense_bias_node.inputs[i]) is None:
            dense_bias_name = dense_bias_node.inputs[i]
            dense_weight_node = graph.get_prev_node(dense_bias_node.inputs[1 - i])
            for j in range(len(dense_weight_node.inputs)):
                if graph.get_prev_node(dense_weight_node.inputs[j]) is None:
                    dense_weight_name = dense_weight_node.inputs[j]
    dense_weight = graph.get_node(dense_weight_name, Initializer)
    dense_bias = graph.get_node(dense_bias_name, Initializer)
    dense_weight.value = dense_weight.value.T
    gemm_dense = graph.add_node(f"gemm_layer{layer}_dense", "Gemm",
                                inputs=[cast_after_flashattention.outputs[0]],
                                outputs=[f"gemm_layer{layer}_dense_out"],
                                attrs={"alpha": 1, "beta": 1, "transB": 1})
    gemm_dense.inputs.extend([dense_weight.name, dense_bias.name])
    addln_node.inputs[0] = gemm_dense.outputs[0]

def add_padinput(graph):
    genseqlen = graph["genseqlen"]
    addln_nodes = graph.get_nodes("Add")
    final_norm = addln_nodes[-2]

    padinput = graph.add_node("padinput", "PadInput")
    graph.insert_node(final_norm.name, padinput)
    gather_batch = graph["batch"]
    gather_maxseqlen = graph["maxseqlen"]
    gather_hidden = graph["hidden"]
    padinput.inputs.extend([genseqlen.outputs[0], gather_batch.outputs[0],
                            gather_maxseqlen.outputs[0], gather_hidden.outputs[0]])
    cast = graph.add_node("cast_after_padinput", "Cast", attrs={"to": 1})
    graph.insert_node(padinput.name, cast)

def add_cast(graph):
    padinput = graph["padinput"]
    cast = graph.add_node(f"cast_before_{padinput.name}", "Cast",
                        attrs={"to": 10})
    graph.insert_node(padinput.name, cast, refer_index=0, mode="before")

def main(graph):

    modify_inputs(graph)

    remove_after_attention_mask_nodes(graph)

    add_genseqlen(graph)

    add_ntokens(graph)

    add_mask(graph)

    add_unpadinput(graph)

    add_gather_bs_max_hidden(graph)

    unpadinput = graph["unpadinput"]
    cast = graph.get_next_nodes(unpadinput.outputs[0])[0]
    modify_self_attention(graph, cast, 0)

    add_nodes = graph.get_nodes("Add")
    layer = 1
    for add_node in add_nodes:
        if len(graph.get_next_nodes(add_node.outputs[0])) == 4:
            next_nodes_type = [node.op_type for node in graph.get_next_nodes(add_node.outputs[0])]
            if "MatMul" in next_nodes_type and "Add" in next_nodes_type:
                modify_self_attention(graph, add_node, layer)
                layer += 1

    add_padinput(graph)

    add_cast(graph)

    graph.remove_unused_nodes()

if __name__=="__main__":
    args = parse_args()
    onnx_graph = OnnxGraph.parse(args.input_file)
    main(onnx_graph)
    onnx_graph.infershape()
    onnx_graph.save(args.output_file)

