# Copyright 2021 Huawei Technologies Co., Ltd
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
from gener_core.mod_modify.onnx_graph import OXGraph
from gener_core.mod_modify.onnx_node import OXNode
import numpy as np
import os
import sys
import onnx

weight_path = sys.argv[1]
input_model = sys.argv[2]
save_model = sys.argv[3]


def weight_name(file_path):
    file_name = list()
    for i in os.listdir(file_path):
        if i.isdigit():
            file_name.append(i)
    return sorted(file_name)

mod = OXGraph(input_model)
weight_name_list = weight_name(weight_path)
for j in range(len(weight_name_list)):
    weight_name_list[j] = weight_path + "/" + weight_name_list[j]

shape_list = [(80, 256), (256, 256), (1, 4096, 768), (1, 4096, 1024), (1, 8192), (1024, 128), (32, 128),
              (1, 4096, 1536), (1, 4096, 1024), (1, 8192)]
matmul_310_input2 = np.fromfile(weight_name_list[0], dtype = np.float32).reshape((80, 256))
matmul_311_input2 = np.fromfile(weight_name_list[1], dtype = np.float32).reshape((256, 256))

lstm_329_input2 = np.fromfile(weight_name_list[2], dtype = np.float32).reshape((1, 4096, 768))
lstm_330_input3 = np.fromfile(weight_name_list[3], dtype = np.float32).reshape((1, 4096, 1024))
lstm_331_input4 = np.fromfile(weight_name_list[4], dtype = np.float32).reshape((1, 8192))

matmul_332_input2 = np.fromfile(weight_name_list[5], dtype = np.float32).reshape((1024, 128))
matmul_333_input2 = np.fromfile(weight_name_list[6], dtype = np.float32).reshape((32, 128))

lstm_352_input2 = np.fromfile(weight_name_list[7], dtype = np.float32).reshape((1, 4096, 1536))
lstm_353_input3 = np.fromfile(weight_name_list[8], dtype = np.float32).reshape((1, 4096, 1024))
lstm_354_input4 = np.fromfile(weight_name_list[9], dtype = np.float32).reshape((1, 8192))


conv_input2 = np.fromfile(weight_path + "/" +
                          "tacotron2.decoder.attention_layer.location_layer.location_conv.conv.weight",
                          dtype = np.float32).reshape((32, 2, 31))
gemm1_input2 = np.fromfile(weight_path + "/" +
                          "tacotron2.decoder.linear_projection.linear_layer.weight",
                           dtype = np.float32).reshape((80, 1536))
gemm2_input2 = np.fromfile(weight_path + "/" +
                          "tacotron2.decoder.gate_layer.linear_layer.weight",
                           dtype = np.float32).reshape((1, 1536))

const_node_310 = mod.add_const_node("matmul_310_input2", matmul_310_input2)
const_node_311 = mod.add_const_node("matmul_311_input2", matmul_311_input2)
const_node_329 = mod.add_const_node("lstm_329_input2", lstm_329_input2)
const_node_330 = mod.add_const_node("lstm_330_input3", lstm_330_input3)
const_node_331 = mod.add_const_node("lstm_331_input4", lstm_331_input4)

const_node_332 = mod.add_const_node("matmul_332_input2", matmul_332_input2)
const_node_333 = mod.add_const_node("matmul_333_input2", matmul_333_input2)

const_node_352 = mod.add_const_node("lstm_352_input2", lstm_352_input2)
const_node_353 = mod.add_const_node("lstm_353_input3", lstm_353_input3)
const_node_354 = mod.add_const_node("lstm_354_input4", lstm_354_input4)

const_node_conv = mod.add_const_node("conv_input2", conv_input2)

const_node_gemm1 = mod.add_const_node("gemm1_input2", gemm1_input2)
const_node_gemm2 = mod.add_const_node("gemm2_input2", gemm2_input2)

Conv = mod.get_nodes_by_optype("Conv")
for conv in Conv:
    now_conv_node = mod.get_node(conv.input_name[1])
    if now_conv_node.op_type == "Initializer" or now_conv_node.op_type == "Constant":
        if now_conv_node._node.dims == [32, 2, 31]:
            conv.set_input_node(1, [const_node_conv])

Gemm = mod.get_nodes_by_optype("Gemm")
for gemm in Gemm:
    now_gemm_node = mod.get_node(gemm.input_name[1])
    if now_gemm_node.op_type == "Initializer" or now_gemm_node.op_type == "Constant":
        if now_gemm_node._node.dims == [80, 1536]:
            gemm.set_input_node(1, [const_node_gemm1])
        if now_gemm_node._node.dims == [1, 1536]:
            gemm.set_input_node(1, [const_node_gemm2])

MatMul = mod.get_nodes_by_optype("MatMul")
for matmul in MatMul:
    now_mm_node = mod.get_node(matmul.input_name[1])
    if now_mm_node.op_type == "Initializer" or now_mm_node.op_type == "Constant":
        if now_mm_node._node.dims == [80, 256]:
            matmul.set_input_node(1, [const_node_310])

        if now_mm_node._node.dims == [256, 256]:
            matmul.set_input_node(1, [const_node_311])
        if now_mm_node._node.dims == [1024, 128]:
            matmul.set_input_node(1, [const_node_332])
        if now_mm_node._node.dims == [32, 128]:
            matmul.set_input_node(1, [const_node_333])

LSTM = mod.get_nodes_by_optype("LSTM")
for lstm in LSTM:
    now_lstm_node2 = mod.get_node(lstm.input_name[1])
    now_lstm_node3 = mod.get_node(lstm.input_name[2])
    now_lstm_node4 = mod.get_node(lstm.input_name[3])
    if (now_lstm_node2.op_type == "Initializer" or now_lstm_node2.op_type == "Constant") \
        and (now_lstm_node3.op_type == "Initializer" or now_lstm_node3.op_type == "Constant") \
        and (now_lstm_node4.op_type == "Initializer" or now_lstm_node4.op_type == "Constant"):
            if now_lstm_node2._node.dims == [1, 4096, 768] and now_lstm_node3._node.dims == [1, 4096, 1024] and \
                    now_lstm_node4._node.dims == [1, 8192]:
                lstm.set_input_node(1, [const_node_329, const_node_330, const_node_331])
            if now_lstm_node2._node.dims == [1, 4096, 1536] and now_lstm_node3._node.dims == [1, 4096, 1024] and \
                    now_lstm_node4._node.dims == [1, 8192]:
                lstm.set_input_node(1, [const_node_352, const_node_353, const_node_354])
mod.save_new_model(save_model)