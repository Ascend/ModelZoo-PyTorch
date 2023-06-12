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

import os
import onnx
import onnxruntime
import numpy as np
from auto_optimizer import OnnxNode


def keep_dynamic_batch(graph):
    for node in graph.inputs + graph.outputs:
        node.shape[0] = 'batch'
    graph.infershape()


def get_np_datatype():
    np_datatype = {
        "1": np.float32,
        "2": np.uint8,
        "3": np.int8,
        "4": np.uint16,
        "5": np.int16,
        "6": np.int32,
        "7": np.int64,
        "9": np.bool,
        "10": np.float16,
        "11": np.float64,
        "12": np.uint32,
        "13": np.uint64,
    }
    return np_datatype

def create_mask(graph, input_node):
    # build mask block: slice->squeeze->equal->cast->mul
    if isinstance(input_node, OnnxNode):
        input_name = input_node.outputs[0]
    else:
        input_name = input_node.name
    slice_start = graph.add_initializer(
        "Slice_start",
        np.array([0], dtype="int64")
    )
    slice_end = graph.add_initializer(
        "Slice_end",
        np.array([1], dtype="int64")
    )
    slice_axis = graph.add_initializer(
        "Slice_axis",
        np.array([-1], dtype="int64")
    )
    slice_node = graph.add_node(
        "Slice_mask",
        "Slice",
        inputs=[input_name, slice_start.name, slice_end.name, slice_axis.name],
        outputs=["out_Slice_mask"]
    )
    squeeze_node = graph.add_node(
        "Squeeze_mask",
        "Squeeze",
        inputs=slice_node.outputs,
        attrs={
            "axes": [-1]
        },
        outputs=["out_Squeeze_mask"]
    )
    equal_init = graph.add_initializer(
        "Equal_value",
        np.array(0, dtype="float32")
    )
    equal_node = graph.add_node(
        "Equal_mask",
        "Equal",
        inputs=squeeze_node.outputs + [equal_init.name],
        outputs=["out_Equal_mask"]
    )
    cast_node = graph.add_node(
        "Cast_mask",
        "Cast",
        attrs={
            'to': 1
        },
        inputs=equal_node.outputs,
        outputs=["out_Cast_mask"]
    )
    mul_init = graph.add_initializer(
        "Mul_value",
        np.array(-65504, dtype="float32")
    )
    mul_node = graph.add_node(
        "Mul_mask",
        "Mul",
        inputs=cast_node.outputs + [mul_init.name],
        outputs=["out_Mul_mask"]
    )
    return mul_node


class GraphFusion():
    def __init__(self, input_model, output_model, opt_type=2):
        opt_model_path = input_model + ".optimized.onnx"
        sess_option = onnxruntime.SessionOptions()
        sess_option.optimized_model_filepath = opt_model_path
        sess_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        _ = onnxruntime.InferenceSession(input_model, sess_option, providers=["CPUExecutionProvider"])
        self.model = onnx.load(opt_model_path)
        self.model = onnx.helper.make_model(self.model.graph, opset_imports=[onnx.helper.make_opsetid(domain="", version=11)])
        os.remove(opt_model_path)
        self.output_model = output_model
        self.graph = self.model.graph
        self.nodes = self.graph.node


        self.input = []
        self.output_k = []
        self.output_v = []
        self.kv_pair = {}


        # opt_type = 1: onstant folding only
        # opt_type = 2 :constant folding + kv fusion
        if opt_type == 2:
            self.find_kv_pair()
            self.do_kv_fusion()

        self.fix_unsupported_gather()
        self.save_output_model()

    def save_output_model(self):
        onnx.save(self.model, self.output_model)

    def find_kv_pair(self):
        count_k = 0
        count_v = 0
        for node in self.nodes:
            if node.op_type == "MatMul":
                if "linear_k" in node.name:
                    key = "/".join(node.name.split("/")[1:3])
                    count_k = count_k + 1
                    if key not in self.kv_pair.keys():
                        self.kv_pair[key] = [node.name]
                    else:
                        self.kv_pair[key] = [node.name, self.kv_pair[key][0]]
                elif "linear_v" in node.name:
                    key = "/".join(node.name.split("/")[1:3])
                    count_v = count_v + 1
                    if key not in self.kv_pair.keys():
                        self.kv_pair[key] = [node.name]
                    else:
                        self.kv_pair[key] = [self.kv_pair[key][0], node.name]
        
    def find_matmul_node_index(self, node_name):
        for idx, node in enumerate(self.nodes):
            if node.name == node_name:
                return idx
        return None

    def get_initializer_index(self, weight_name):
        for idx, weight in enumerate(self.graph.initializer):
            if weight.name == weight_name:
                return idx
        return None

    def get_initializer_to_array(self, weight_idx):
        initializer = self.graph.initializer[weight_idx]
        data_type = get_np_datatype().get(str(initializer.data_type))
        weight = np.frombuffer(initializer.raw_data, dtype=data_type)
        if weight.size == 0:
            weight = np.array(initializer.float_data)
        weight = weight.reshape(initializer.dims)
        return weight

    def concat_matmul_weight(self, k_weight, v_weight, weight_name):
        weight = np.concatenate((k_weight.T, v_weight.T), 0).T
        matmul_weight = onnx.helper.make_tensor(
            weight_name,
            onnx.TensorProto.FLOAT,
            weight.shape,
            weight.tobytes(),
            raw=True
        )
        return matmul_weight

    def concat_bias_weight(self, k_bias, v_bias, bias_name):
        bias = np.concatenate((k_bias, v_bias), 0)
        bias_weight = onnx.helper.make_tensor(
            bias_name,
            onnx.TensorProto.FLOAT,
            bias.shape,
            bias.tobytes(),
            raw=True
        )
        return bias_weight
    
    def do_kv_fusion(self):
        for value in self.kv_pair.values():
            k_name = value[0]
            v_name = value[1]
            k_node_idx = self.find_matmul_node_index(k_name)
            v_node_idx = self.find_matmul_node_index(v_name)

            k_weight_idx = self.get_initializer_index(self.nodes[k_node_idx].input[1])
            k_weight = self.get_initializer_to_array(k_weight_idx)
            k_add_idx = self.find_node_index(self.nodes[k_node_idx].output[0], "Add")
            k_add_weight_idx = self.get_initializer_index(self.nodes[k_add_idx].input[0])
            k_add_weight = self.get_initializer_to_array(k_add_weight_idx)

            v_weight_idx = self.get_initializer_index(self.nodes[v_node_idx].input[1])
            v_weight = self.get_initializer_to_array(v_weight_idx)
            v_add_idx = self.find_node_index(self.nodes[v_node_idx].output[0], "Add")
            v_add_weight_idx = self.get_initializer_index(self.nodes[v_add_idx].input[0])
            v_add_weight = self.get_initializer_to_array(v_add_weight_idx)

            matmul_weight = self.concat_matmul_weight(k_weight, v_weight, self.nodes[k_node_idx].input[1])
            self.graph.initializer.append(matmul_weight)
            bias_name = self.nodes[k_add_idx].input[0]
            bias_weight = self.concat_bias_weight(k_add_weight, v_add_weight, bias_name)
            self.graph.initializer.append(bias_weight)

            self.input = [self.nodes[k_node_idx].input[0], self.nodes[k_node_idx].input[1]]
            self.output_k = self.delete_old_nodes(k_node_idx)
            v_node_idx = self.find_matmul_node_index(v_name)
            self.output_v = self.delete_old_nodes(v_node_idx)
            self.insert_new_nodesv2(v_node_idx, v_name, bias_name)

    def find_node_index(self, input_tensor, node_type):
        for idx, node in enumerate(self.nodes):
            if node.op_type == node_type and input_tensor in node.input:
                return idx
        return None

    def delete_old_nodes(self, node_idx):
        node = self.nodes[node_idx]
        node_output = self.nodes[node_idx].output[0]
        self.nodes.remove(node)

        # delete add
        add_index = self.find_node_index(node_output, "Add")
        add_output = self.nodes[add_index].output[0]
        self.nodes.remove(self.nodes[add_index])

        # delete reshape
        reshape_index = self.find_node_index(add_output, "Reshape")
        reshape_output = self.nodes[reshape_index].output[0]
        self.nodes.remove(self.nodes[reshape_index])

        # detele transpose
        transpose_index = self.find_node_index(reshape_output, "Transpose")
        output = [self.nodes[transpose_index].output[0]]
        self.nodes.remove(self.nodes[transpose_index])
        return output
    
    def insert_new_nodesv2(self, index, name, bias_name):
        node_lists = []
        matmul_output = [name + "/matmul"]
        matmul_node = onnx.helper.make_node(inputs=self.input, outputs=matmul_output, op_type="MatMul")
        node_lists.append(matmul_node)

        add_output = [name + "/add"]
        add_node = onnx.helper.make_node(inputs=matmul_output + [bias_name], outputs=add_output, op_type="Add")
        node_lists.append(add_node)

        reshape_output = [name + "/reshape"]
        reshape_dim_name = name + "/reshape_dim"
        reshape_dim = onnx.helper.make_tensor(
            reshape_dim_name,
            onnx.TensorProto.INT64,
            (5,), np.array([0, 0, 2, 4, 64])
        )
        self.graph.initializer.append(reshape_dim)
        reshape_node = onnx.helper.make_node(inputs=[add_output[0], reshape_dim_name], outputs=reshape_output, op_type="Reshape")
        node_lists.append(reshape_node)

        transpose_output = [name + "/transpose"]
        transpose_node1 = onnx.helper.make_node(inputs=reshape_output, outputs=transpose_output, op_type="Transpose", perm=[2, 0, 3, 1, 4])
        node_lists.append(transpose_node1)

        gather3_indices_name = name + "/gather3_indices"
        gather3_indices = onnx.helper.make_tensor(
            gather3_indices_name,
            onnx.TensorProto.INT64,
            (), np.array([1]))
        self.graph.initializer.append(gather3_indices)
        gather_node3 = onnx.helper.make_node(inputs=transpose_output + [gather3_indices_name], outputs=self.output_v, op_type="Gather", axis=0)
        node_lists.append(gather_node3)

        gather4_output = [name + "/gather4"]
        gather4_indices_name = name + "/gather4_indices"
        gather4_indices = onnx.helper.make_tensor(
            gather4_indices_name,
            onnx.TensorProto.INT64,
            (), np.array([0]))
        self.graph.initializer.append(gather4_indices)
        gather_node4 = onnx.helper.make_node(inputs=transpose_output + [gather4_indices_name], outputs=gather4_output, op_type="Gather", axis=0)
        node_lists.append(gather_node4)

        transpose_node2 = onnx.helper.make_node(inputs=gather4_output, outputs=self.output_k, op_type="Transpose", perm=[0, 1, 3, 2])
        node_lists.append(transpose_node2)

        for idx, item in enumerate(node_lists):
            self.nodes.insert(index + idx, item)
        
    def is_unsupported(self, node):
        indices = node.input[1]
        for init in self.graph.initializer:
            if init.name == indices:
                data_type = get_np_datatype().get(str(init.data_type))
                out = np.frombuffer(init.raw_data, dtype=data_type)
                if out == -1:
                    return True
        return False
    
    def fix_unsupported_gather(self):
        for idx, node in enumerate(self.nodes):
            if node.op_type == "Gather":
                if self.is_unsupported(node):
                    print("fix gather node:", node.name)

                    # shape
                    shape_output = self.nodes[idx].output[0] + "/shape"
                    shape_node = onnx.helper.make_node(inputs=[self.nodes[idx].input[0]], outputs=[shape_output], op_type="Shape")

                    # gather
                    gather_output = self.nodes[idx].output[0] + "/gather1"
                    gather_indices_name = self.nodes[idx].input[1] + "/gather_indices"
                    gather_indices = onnx.helper.make_tensor(
                        gather_indices_name,
                        onnx.TensorProto.INT64,
                        (), np.array([1]))
                    self.graph.initializer.append(gather_indices)
                    gather_node = onnx.helper.make_node(inputs=[shape_output, gather_indices_name], outputs=[gather_output], op_type="Gather", axis=0)

                    # sub
                    sub_output = self.nodes[idx].output[0] + "/sub"
                    sub_input_name = self.nodes[idx].output[0] + "/sub_input"
                    sub_input = onnx.helper.make_tensor(
                        sub_input_name,
                        onnx.TensorProto.INT64,
                        (), np.array([1]))
                    self.graph.initializer.append(sub_input)
                    sub_node = onnx.helper.make_node(inputs=[gather_output, sub_input_name], outputs=[sub_output], op_type="Sub")

                    # gather2
                    gather_node2 = onnx.helper.make_node(inputs=[self.nodes[idx].input[0], sub_output], outputs=self.nodes[idx].output, op_type="Gather", axis=1)

                    self.nodes.remove(self.nodes[idx])

                    self.nodes.insert(idx, shape_node)
                    self.nodes.insert(idx + 1, gather_node)
                    self.nodes.insert(idx + 2, sub_node)
                    self.nodes.insert(idx + 3, gather_node2)
