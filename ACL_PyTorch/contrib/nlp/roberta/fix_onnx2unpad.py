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
    parser = argparse.ArgumentParser(description="fix roberta onnx")
    parser.add_argument("--input_file", type=str, required=True,
                        help="path to pth model")
    parser.add_argument("--output_file", type=str, required=True,
                        help="path to save onnx model")
    arg = parser.parse_args()
    return arg

def get_config(graph):
    input_ph = graph.inputs[0]
    bs, seq_len = input_ph.shape[0], input_ph.shape[1]
    return bs, seq_len

def add_genseqlen(graph):
    input_ph = graph.inputs[0]
    nodes_after_input = graph.get_next_nodes(input_ph.name)
    for node in nodes_after_input:
        if node.op_type == "Equal":
            if len(graph.get_next_nodes(node.outputs[0])) > 1:
                not_node = graph.add_node("not_of_equal", "Not", inputs=[node.outputs[0]], outputs=[f"{node.name}.1"])
                cast = graph.add_node(f"cast_after_{not_node.name}", "Cast",
                                      inputs=[not_node.outputs[0]], outputs=[f"{not_node.name}.1"],
                                      attrs={"to": 6})
                genseqlen = graph.add_node("genseqlen", "GenSeqLen",
                                           inputs=[cast.outputs[0]], outputs=["seqlen", "seqlen_ori"])
                
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

def add_unpadinput(graph, bs, max_seqlen):
    genseqlen = graph["genseqlen"]
    mul_nodes = graph.get_nodes("Mul")
    ntokens = graph["ntokens"]
    for mul_node in mul_nodes:
        if len(graph.get_next_nodes(mul_node.outputs[0])) == 4:
            reshape = graph.add_node(f"reshape_after_{mul_node.name}", "Reshape")
            graph.insert_node(mul_node.name, reshape, refer_index=0, mode="after")
            reshape_ini = graph.add_initializer("reshape_dim", value=np.array([bs, max_seqlen, 768], dtype=np.int64))
            reshape.inputs.append(reshape_ini.name)
            cast = graph.add_node("cast_after_input_embedding", "Cast", attrs={"to": 10})
            graph.insert_node(reshape.name, cast, refer_index=0, mode="after")
            unpadinput = graph.add_node("unpadinput", "UnpadInput")
            graph.insert_node(cast.name, unpadinput, refer_index=0, mode="after")
            unpadinput.inputs.extend([genseqlen.outputs[0], ntokens.outputs[0]])
            cast = graph.add_node("cast_after_unpadinput", "Cast",
                                  inputs=[unpadinput.outputs[0]], outputs=[f"{unpadinput.name}.1"],
                                  attrs={"to": 1})
            graph.insert_node(unpadinput.name, cast, refer_index=0, mode="after")

def add_gather_bs_max_hidden(graph):
    src_tokens = graph.inputs[0]
    nodes_after_input = graph.get_next_nodes(src_tokens.name)
    for node in nodes_after_input:
        if node.op_type == "Gather":
            shape_node1 = graph.add_node("shape_of_gather1", "Shape",
                                         inputs=[node.outputs[0]], outputs=["shape_of_gather_out1"])
            shape_node2 = graph.add_node("shape_of_gather2", "Shape",
                                         inputs=[node.outputs[0]], outputs=["shape_of_gather_out2"])
            shape_node3 = graph.add_node("shape_of_gather3", "Shape",
                                         inputs=[node.outputs[0]], outputs=["shape_of_gather_out3"])

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
            break

def add_padinput(graph):
    genseqlen = graph["genseqlen"]
    addln_nodes = graph.get_nodes("Add")
    final_norm = addln_nodes[-1]
    padinput = graph.add_node("padinput", "PadInput")
    graph.insert_node(final_norm.name, padinput)
    gather_batch = graph["batch"]
    gather_maxseqlen = graph["maxseqlen"]
    gather_hidden = graph["hidden"]

    padinput.inputs.extend([genseqlen.outputs[0], gather_batch.outputs[0],
                            gather_maxseqlen.outputs[0], gather_hidden.outputs[0]])
    cast = graph.add_node("cast_after_padinput", "Cast", attrs={"to": 1})
    graph.insert_node(padinput.name, cast)

def modify_self_attention(graph, input_node, layer):
    genseqlen = graph["genseqlen"]
    ntokens = graph["ntokens"]
    nodes4 = graph.get_next_nodes(input_node.outputs[0])

    weight_bias = {}
    matmul_name = []  # 记录matmul节点的name，后续需要删掉
    for node in nodes4:
        if node.op_type == "Add":
            addln_node = node
            addln_inputs = node.inputs
            dense_bias_node = graph.get_prev_node(addln_inputs[1])
        if node.op_type == "MatMul":
            matmul_name.append(node.name)
            bias = graph.get_next_nodes(node.outputs[0])[0]
            reshape = graph.get_next_nodes(bias.outputs[0])[0]
            transpose = graph.get_next_nodes(reshape.outputs[0])[0]
            node_after_transpose = graph.get_next_nodes(transpose.outputs[0])[0]
            if node_after_transpose.op_type == transpose.op_type:
                # 相等说明此时在key的分支
                reshapeK = graph.get_prev_node(transpose.inputs[0])
                biasK = graph.get_prev_node(reshapeK.inputs[0])
                weightK = graph.get_prev_node(biasK.inputs[0])
                weight_bias["key"] = [weightK.inputs[1], biasK.inputs[1]]
            else:
                # 不相等的话，通过matmul的上游节点是不是同种类型来区分q和v
                node1 = graph.get_prev_node(node_after_transpose.inputs[0])
                node2 = graph.get_prev_node(node_after_transpose.inputs[1])
                if node1.op_type == node2.op_type:
                    # node1和node2种类相等说明这个matmul是QK矩阵乘
                    reshapeQ = graph.get_prev_node(node1.inputs[0])
                    biasQ = graph.get_prev_node(reshapeQ.inputs[0])
                    weightQ = graph.get_prev_node(biasQ.inputs[0])
                    weight_bias["query"] = [weightQ.inputs[1], biasQ.inputs[1]]
                else:
                    weightV = node
                    biasV = bias
                    weight_bias["value"] = [weightV.inputs[1], biasV.inputs[1]]
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
                                               inputs=[flashattention.outputs[0]], outputs=[f"{flashattention.name}.1"],
                                               attrs={"to": 1})

    dense_bias = dense_bias_node
    dense_weight = graph.get_prev_node(dense_bias_node.inputs[0])
    dense_weight.inputs[0] = cast_after_flashattention.outputs[0]

def add_cast(graph):
    padinput = graph["padinput"]
    cast = graph.add_node(f"cast_before_{padinput.name}", "Cast",
                          attrs={"to": 10})
    graph.insert_node(padinput.name, cast, refer_index=0, mode="before")

def main(graph):
    bs, max_seqlen = get_config(graph)

    add_genseqlen(graph)

    add_ntokens(graph)

    add_mask(graph)

    add_unpadinput(graph, bs, max_seqlen)

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