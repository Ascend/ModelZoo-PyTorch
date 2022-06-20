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

import argparse
import numpy as np
import copy
from gener_core.mod_modify.interface import AttrType as AT
from gener_core.mod_modify.onnx_graph import OXGraph

DATA_SZ = 21490
SPLIT_NUM = 5

class sub_strucure(object):
    def __init__(self, mod, conv1_node, selu_node, conv2_node, conv3_node, add_node):
        self.mod = mod
        self.conv1_node = conv1_node
        self.selu_node = selu_node
        self.conv2_node = conv2_node
        self.conv3_node = conv3_node
        self.add_node = add_node
    
    def create_slice_node(self, sub_data_idx, sub_data_sz):
        begin = max(sub_data_sz * sub_data_idx - 1, 0)
        end = min(sub_data_sz * (sub_data_idx + 1) + 1, DATA_SZ) 
        begin_node = self.mod.add_const_node(f"const_begin_{sub_data_idx}", np.array([begin], np.int32))
        end_node = self.mod.add_const_node(f"const_end_{sub_data_idx}", np.array([end], np.int32))
        axes_node = self.mod.add_const_node(f"const_axes_{sub_data_idx}", np.array([3], np.int32))
        step_node = self.mod.add_const_node(f"const_step_{sub_data_idx}", np.array([1], np.int32))
        s_node = self.mod.add_new_node(f"Slice_{sub_data_idx}", "Slice")
        return s_node, begin_node, end_node, axes_node, step_node

    def create_conv_node(self, conv_node, sub_data_idx, update_pad):
        pad = conv_node.get_attr("pads", AT.LIST_INT)
        if update_pad:
            if sub_data_idx == 0:
                pad[3] = 0
            elif sub_data_idx == (SPLIT_NUM-1):
                pad[1] = 0
            else:
                pad[1], pad[3] = 0, 0
        c_node = self.mod.add_new_node(f"{conv_node.name}_{sub_data_idx}", "Conv",
                                  {"dilations": (AT.LIST_INT, [1, 1]),
                                   "group": (AT.INT, 1),
                                   "kernel_shape": (AT.LIST_INT, conv_node.get_attr("kernel_shape", AT.LIST_INT)),
                                   "pads": (AT.LIST_INT, pad),
                                   "strides": (AT.LIST_INT, [1, 1])})
        w_node = self.mod.get_node(conv_node.input_name[1])
        b_node = self.mod.get_node(conv_node.input_name[2])
        return c_node, w_node, b_node

    def create_selu_node(self, selu_node, sub_data_idx):
        s_node = self.mod.add_new_node(f"Selu_{sub_data_idx}", "Selu")
        return s_node

    def create_add_node(self, add_node, sub_data_idx):
        a_node = self.mod.add_new_node(f"Add_{sub_data_idx}", "Add")
        return a_node

    def create_sub_structure(self, x_node, sub_data_idx, sub_data_sz):
        slice_node, begin_node, end_node, axes_node, step_node = self.create_slice_node(sub_data_idx, sub_data_sz)
        slice_node.set_input_node(0, [x_node, begin_node, end_node, axes_node, step_node])

        conv1_node, weight1_node, bias1_node = self.create_conv_node(self.conv1_node, sub_data_idx, True)
        conv1_node.set_input_node(0, [slice_node, weight1_node, bias1_node])
    
        selu_node = self.create_selu_node(self.selu_node, sub_data_idx)
        selu_node.set_input_node(0, [conv1_node])

        conv2_node, weight2_node, bias2_node = self.create_conv_node(self.conv2_node, sub_data_idx, False)
        conv2_node.set_input_node(0, [selu_node, weight2_node, bias2_node])

        conv3_node, weight3_node, bias3_node = self.create_conv_node(self.conv3_node, sub_data_idx, True)
        conv3_node.set_input_node(0, [slice_node, weight3_node, bias3_node])

        add_node = self.create_add_node(self.add_node, sub_data_idx)
        add_node.set_input_node(0, [conv2_node, conv3_node])

        return add_node


class sub_gather_strucure(object):
    def __init__(self, mod):
        self.mod = mod
    
    def create_slice_node(self, sub_axis, gather_n, input_n):
        begin = sub_axis
        end = sub_axis + 1
        begin_node = self.mod.add_const_node(f"const_begin_s{sub_axis}_g{gather_n}_i{input_n}", np.array([begin], np.int32))
        end_node = self.mod.add_const_node(f"const_end_s{sub_axis}_g{gather_n}_i{input_n}", np.array([end], np.int32))
        axes_node = self.mod.add_const_node(f"const_axes_s{sub_axis}_g{gather_n}_i{input_n}", np.array([0], np.int32))
        step_node = self.mod.add_const_node(f"const_step_s{sub_axis}_g{gather_n}_i{input_n}", np.array([1], np.int32))
        s_node = self.mod.add_new_node(f"Slice_s{sub_axis}_g{gather_n}_i{input_n}", "Slice")
        return s_node, begin_node, end_node, axes_node, step_node
    
    def create_cast_node(self, sub_axis, gather_n):
        c_node = self.mod.add_new_node(f"Cast_a{sub_axis}_g{gather_n}", "Cast",
                                       {"to": (AT.LIST_INT, 6)})
        return c_node

    def create_flatten_node(self, sub_axis, gather_n):
        f_node = self.mod.add_new_node(f"Flatten_a{sub_axis}_g{gather_n}", "Flatten",
                                       {"axis": (AT.INT, 1)})
        return f_node

    def create_gather_node(self, sub_axis, gather_n):
        g_node = self.mod.add_new_node(f"Gather_a{sub_axis}_g{gather_n}", "Gather",
                                       {"axis": (AT.INT, 1)})
        return g_node

    def create_sub_structure(self, x1_node, x2_node, sub_axis, gather_n):
        slice_node1, begin_node, end_node, axes_node, step_node = self.create_slice_node(sub_axis, gather_n, 1)
        slice_node1.set_input_node(0, [x1_node, begin_node, end_node, axes_node, step_node])

        cast_node = self.create_cast_node(sub_axis, gather_n)
        cast_node.set_input_node(0, [slice_node1])

        slice_node2, begin_node, end_node, axes_node, step_node = self.create_slice_node(sub_axis, gather_n, 2)
        slice_node2.set_input_node(0, [x2_node, begin_node, end_node, axes_node, step_node])

        flatten_node = self.create_flatten_node(sub_axis, gather_n)
        flatten_node.set_input_node(0, [cast_node])
    
        gather_node = self.create_gather_node(sub_axis, gather_n)
        gather_node.set_input_node(0, [slice_node2, flatten_node])

        return gather_node


def data_slice(mod):
    conv1_node = mod.get_node("Conv_8")
    selu_node = mod.get_node("Selu_9")
    conv2_node = mod.get_node("Conv_10")
    conv3_node = mod.get_node("Conv_11")
    add_node = mod.get_node("Add_12")
    sub_s = sub_strucure(mod, conv1_node, selu_node, conv2_node, conv3_node, add_node)
    x_node = mod.get_node(conv1_node.input_name[0])

    np.random.seed(1737)
    SUB_DATA_SZ = DATA_SZ // SPLIT_NUM

    # conv_structure split
    add_nodes = []
    for i in range(SPLIT_NUM):
        a_node = sub_s.create_sub_structure(x_node, i, SUB_DATA_SZ)
        add_nodes.append(a_node)

    concat_node = mod.add_new_node(f"Concat_{np.random.randint(0, 7393)}", "Concat",
                                   {"axis": (AT.INT, -1)})
    concat_node.set_input_node(0, add_nodes)

    # relink
    maxpool_node = mod.get_node("MaxPool_13")
    maxpool_node.set_input_node(0, [concat_node])

    # remove ori node
    mod.node_remove([conv1_node.name, selu_node.name, conv2_node.name, conv3_node.name, add_node.name])
    

def split_conv(mod):
    unsqueeze1_node = mod.get_node("Unsqueeze_0")
    conv_node = mod.get_node("Conv_2")
    unsqueeze2_node = mod.get_node("Unsqueeze_3")
    abs_node = mod.get_node("Abs_4")
    c_node = mod.add_new_node(f"{conv_node.name}_0", "Conv",
                              {"dilations": (AT.LIST_INT, [1, 1]),
                              "group": (AT.INT, 1),
                              "kernel_shape": (AT.LIST_INT, [70, 129]),
                              "pads": (AT.LIST_INT, [0, 0, 0, 0]),
                              "strides": (AT.LIST_INT, [1, 1])})
    x_node = mod.get_node(unsqueeze1_node.input_name[0])
    input_node = mod.add_placeholder_node("input_0", "float32", [1, 1, 70, 64600])
    weight_node = mod.get_node(conv_node.input_name[1])
    weight_value = np.array(weight_node.const_value).reshape(1,1,70,129)
    print(weight_value)
    w_node = mod.add_const_node("const_weight_0", np.array(weight_value, np.float32))
    c_node.set_input_node(0, [input_node, w_node])
    abs_node.set_input_node(0, [c_node])
    mod.node_remove([x_node.name, unsqueeze1_node.name, conv_node.name, weight_node.name, unsqueeze2_node.name])


def exchange_unsqueeze(mod):
    conv_node = mod.get_node("Conv_2")
    unsqueeze_node = mod.get_node("Unsqueeze_3")
    abs_node = mod.get_node("Abs_4")
    maxpool_node = mod.get_node("MaxPool_5")
    abs_node.set_input_node(0, [conv_node])
    unsqueeze_node.set_input_node(0, [abs_node])
    maxpool_node.set_input_node(0, [unsqueeze_node])


def extend_conv(mod, batch_sizes):
    unsqueeze1_node = mod.get_node("Unsqueeze_0")
    unsqueeze1_node.set_attr({"axes": (AT.LIST_INT, [1, 2])})
    conv_node = mod.get_node("Conv_2")
    conv_node.set_attr({"dilations": (AT.LIST_INT, [1, 1])})
    conv_node.set_attr({"kernel_shape": (AT.LIST_INT, [1, 129])})
    conv_node.set_attr({"pads": (AT.LIST_INT, [0, 0, 0, 0])})
    conv_node.set_attr({"strides": (AT.LIST_INT, [1, 1])})
    weight_node = mod.get_node(conv_node.input_name[1])
    # weight_value = weight_node.const_value
    weight_value = weight_node.const_value[:69, :, :]
    new_weight_value = np.expand_dims(weight_value, axis=1)
    new_weight_node = mod.add_const_node("const_weight", new_weight_value)
    mod.node_replace(weight_node, new_weight_node)

    unsqueeze2_node = mod.get_node("Unsqueeze_3")
    abs_node = mod.get_node("Abs_4")
    abs_node.set_input_node(0, [conv_node])
    mod.node_remove([unsqueeze2_node.name])

    maxpool_node = reshape_maxpool(mod, batch_sizes)


def split_maxpool(mod):
    maxpool1_node = mod.get_node("MaxPool_5")
    maxpool1_node.set_attr({"kernel_shape": (AT.LIST_INT, [1, 3])})
    maxpool1_node.set_attr({"strides": (AT.LIST_INT, [1, 3])})
    
    transpose_node = mod.add_new_node(f"Transpose_0", "Transpose",
                                  {"perm": (AT.LIST_INT, [0, 2, 1, 3])})
    transpose_node.set_input_node(0, [maxpool1_node])

    maxpool2_node = mod.add_new_node(f"MaxPool_0", "MaxPool",
                                  {"ceil_mode": (AT.INT, 0),
                                   "kernel_shape": (AT.LIST_INT, [3, 1]),
                                   "pads": (AT.LIST_INT, [0, 0, 0, 0]),
                                   "strides": (AT.LIST_INT, [3, 1])})
    maxpool2_node.set_input_node(0, [transpose_node])

    bn_node = mod.get_node("BatchNormalization_6")
    bn_node.set_input_node(0, [maxpool2_node])


def reshape_maxpool(mod, batch_sizes):
    maxpool1_node = mod.get_node("MaxPool_5")
    maxpool1_node.set_attr({"kernel_shape": (AT.LIST_INT, [1, 3])})
    maxpool1_node.set_attr({"strides": (AT.LIST_INT, [1, 3])})
    
    reshape_node = mod.add_new_node(f"Reshape_0", "Reshape")
    shape_node = mod.add_const_node(f"const_shape_0", np.array([batch_sizes, 23, 3, 21490], np.int32))
    reshape_node.set_input_node(0, [maxpool1_node, shape_node])

    maxpool2_node = mod.add_new_node(f"MaxPool_0", "MaxPool",
                                  {"ceil_mode": (AT.INT, 0),
                                   "kernel_shape": (AT.LIST_INT, [3, 1]),
                                   "pads": (AT.LIST_INT, [0, 0, 0, 0]),
                                   "strides": (AT.LIST_INT, [3, 1])})
    maxpool2_node.set_input_node(0, [reshape_node])

    bn_node = mod.get_node("BatchNormalization_6")
    scale_node = mod.get_node(bn_node.input_name[1])
    new_scale_value= scale_node.const_value.repeat(23)
    scale_node.set_const_value(new_scale_value)
    b_node = mod.get_node(bn_node.input_name[2])
    new_b_value= b_node.const_value.repeat(23)
    b_node.set_const_value(new_b_value)
    mean_node = mod.get_node(bn_node.input_name[3])
    new_mean_value= mean_node.const_value.repeat(23)
    mean_node.set_const_value(new_mean_value)
    var_node = mod.get_node(bn_node.input_name[4])
    new_var_value= var_node.const_value.repeat(23)
    var_node.set_const_value(new_var_value)
    bn_node.set_input_node(0, [maxpool2_node, scale_node, b_node, mean_node, var_node])

    selu_node = mod.get_node("Selu_7")
    selu_node.set_input_node(0, [bn_node])

    transpose_node = mod.add_new_node(f"Transpose_0", "Transpose",
                                  {"perm": (AT.LIST_INT, [0, 2, 1, 3])})
    transpose_node.set_input_node(0, [selu_node])

    conv1_node = mod.get_node("Conv_8")
    conv1_node.set_input_node(0, [transpose_node])


def replace_gather(mod, io_map):
    gather_nodes = mod.get_nodes_by_optype("GatherElements")
    for i, g_node in enumerate(gather_nodes):
        expand_node = mod.get_node(g_node.input_name[1])
        flatten_node = mod.add_new_node(f"Flatten_{i}", "Flatten",
                                        {"axis": (AT.INT, 1)})
        flatten_node.set_input_node(0, [expand_node.input_name[0]])
        # squeeze_node1 = mod.add_new_node(f"Squeeze1_{i}", "Squeeze",
        #                           {"axes": (AT.LIST_INT, [2])})
        # squeeze_node1.set_input_node(0, [expand_node.input_name[0]])

        gather_node = mod.add_new_node(f"Gather_{i}", "Gather",
                                       {"axis": (AT.INT, 1)})
        gather_node.set_input_node(0, [g_node.input_name[0], flatten_node])

        squeeze_node2 = mod.add_new_node(f"Squeeze2_{i}", "Squeeze",
                                  {"axes": (AT.LIST_INT, [1])})
        squeeze_node2.set_input_node(0, [gather_node])

        matmul1_node = mod.get_node(io_map.get(g_node.name)[0])
        m1_node = mod.get_node(matmul1_node.input_name[1])
        matmul1_node.set_input_node(0, [squeeze_node2, m1_node])

        matmul2_node = mod.get_node(io_map.get(g_node.name)[1])
        m2_node = mod.get_node(matmul2_node.input_name[1])
        matmul2_node.set_input_node(0, [squeeze_node2, m2_node])

        mod.node_remove([expand_node.name, g_node.name])


def replace_gather_bsn(mod, io_map, batch_sizes):
    gather_nodes = mod.get_nodes_by_optype("GatherElements")
    for i, gather_node in enumerate(gather_nodes):
        expand_node = mod.get_node(gather_node.input_name[1])

        sub_s = sub_gather_strucure(mod)
        x1_node = mod.get_node(expand_node.input_name[0])
        x2_node = mod.get_node(gather_node.input_name[0])
        g_nodes = []
        for batch in range(batch_sizes):
            g_node = sub_s.create_sub_structure(x1_node, x2_node, batch, i)
            g_nodes.append(g_node)

        concat_node = mod.add_new_node(f"Concat_{np.random.randint(0, 7393)}", "Concat",
                                    {"axis": (AT.INT, 0)})
        concat_node.set_input_node(0, g_nodes)

        squeeze_node2 = mod.add_new_node(f"Squeeze2_{i}", "Squeeze",
                                  {"axes": (AT.LIST_INT, [1])})
        squeeze_node2.set_input_node(0, [concat_node])

        matmul1_node = mod.get_node(io_map.get(gather_node.name)[0])
        m1_node = mod.get_node(matmul1_node.input_name[1])
        matmul1_node.set_input_node(0, [squeeze_node2, m1_node])

        matmul2_node = mod.get_node(io_map.get(gather_node.name)[1])
        m2_node = mod.get_node(matmul2_node.input_name[1])
        matmul2_node.set_input_node(0, [squeeze_node2, m2_node])

        mod.node_remove([expand_node.name, gather_node.name])


def replace_scatternd(mod, io_map):
    scatternd_nodes = mod.get_nodes_by_optype("ScatterND")
    for i, scatternd_node in enumerate(scatternd_nodes):
        reshape_node = mod.get_node(scatternd_node.input_name[2])
        expand_node = mod.get_node(reshape_node.input_name[0])
        matmul_node = mod.get_node(expand_node.input_name[0])

        indices_node = mod.get_node(scatternd_node.input_name[1])

        if i % 4 == 0:
            concat1, concat2 = [], []

        if indices_node.const_value[0, 0, 0, 0, 1]:
            concat1.append(matmul_node)
        else:
            concat2.append(matmul_node)
        
        if i % 4 == 3:
            div_node = mod.get_node(io_map.get(scatternd_node.name)[0])

            concat_node1 = mod.add_new_node(f"Concat1_{i}", "Concat",
                                    {"axis": (AT.INT, 2)})
            concat_node1.set_input_node(0, concat1)

            concat_node2 = mod.add_new_node(f"Concat2_{i}", "Concat",
                                    {"axis": (AT.INT, 2)})
            concat_node2.set_input_node(0, concat2)

            concat_node3 = mod.add_new_node(f"Concat_{i}", "Concat",
                                    {"axis": (AT.INT, 1)})
            concat_node3.set_input_node(0, [concat_node1, concat_node2])
            div_node.set_input_node(0, [concat_node3])

        mod.node_remove([reshape_node.name, expand_node.name, scatternd_node.name])


def reduce_softmax(mod, io_map):
    softmax_nodes = mod.get_nodes_by_optype("Softmax")
    for i, s_node in enumerate(softmax_nodes):
        axis_value = s_node.get_attr("axis", AT.INT) - 1
        s_node.set_attr({"axis": (AT.INT, axis_value)})
        b1_node = mod.get_node(s_node.input_name[0])
        b2_node = mod.get_node(b1_node.input_name[0])
        s_node.set_input_node(0, [b2_node])
        a1_node = mod.get_node(io_map.get(s_node.name)[0])
        a2_node = mod.get_node(io_map.get(a1_node.name)[0])
        a2_node.set_input_node(0, [s_node])
        mod.node_remove([b1_node.name, a1_node.name])


def make_model(input_onnx, output_onnx, batch_sizes):
    mod = OXGraph(input_onnx)
    io_map = mod.get_net_in_out_map()

    # split_conv(mod)
    # exchange_unsqueeze(mod)
    extend_conv(mod, batch_sizes)
    data_slice(mod)
    replace_gather(mod, io_map)
    # replace_gather_bsn(mod, io_map, batch_sizes)
    replace_scatternd(mod, io_map)
    reduce_softmax(mod, io_map)
    
    mod.save_new_model(output_onnx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AASIST')
    parser.add_argument('--input-onnx', default="aasist_bs1.onnx", type=str,
                        help='input original onnx')
    parser.add_argument('--output-onnx', default="aasist_bs1.onnx", type=str,
                        help='output modified onnx')
    parser.add_argument('--batch-sizes', default=1, type=int,
                        help='batch sizes')
    args = parser.parse_args()
    make_model(args.input_onnx, args.output_onnx, args.batch_sizes)
    print("modify successfully!")
