# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import operator
import numpy as np
from gener_core.mod_modify.onnx_graph import OXGraph
from gener_core.mod_modify.onnx_node import OXNode
from gener_core.mod_modify.interface import AttrType as AT


# pylint: disable=redefined-outer-name
def remove_one_in_one_out_node(mod, node, before_node, after_node):
    mod.node_remove(node)
    after_node.set_input_node(0, [before_node])


mod = OXGraph("encoder.onnx")
io_map = mod.get_net_in_out_map()

# del some Mul with B = 1
Mul_lists = mod.get_nodes_by_optype("Mul")
del_div_name = []
for mul_node in Mul_lists:
    now_Mul = mod.get_node(mul_node)
    Mul_input2 = mod.get_node(now_Mul.input_name[1])
    if Mul_input2.op_type in (
            "Initializer", "Constant") and Mul_input2.const_value.shape is () and Mul_input2.const_value == 1.0:
        Mul_input1 = mod.get_node(now_Mul.input_name[0])
        now_Mul_after_node = mod.get_node(io_map.get(now_Mul.name)[0])
        now_Mul_after_node.set_input_node(1, [Mul_input1])
        mod.node_remove([now_Mul])

# revise the conv input shape and weight, attr to del transdata in the om
Split_lists = mod.get_nodes_by_optype("Split")
for split_node in Split_lists:
    now_Split = mod.get_node(split_node)
    now_Split_input_conv = mod.get_node(now_Split.input_name[0])
    unsqueeze_node = mod.add_new_node(now_Split_input_conv.name + "unsqueeze", "Unsqueeze",
                                      {"axes": (AT.LIST_INT, [2])
                                       })
    now_Split_input_conv_before_node = mod.get_node(now_Split_input_conv.input_name[0])
    now_Split_input_conv.set_input_node(0, [unsqueeze_node])
    unsqueeze_node.set_input_node(0, [now_Split_input_conv_before_node])
    # change the weight/attr of the conv, weight shape 512,256,1 to 512,256,1,1, attr
    now_conv_weight_node = mod.get_node(now_Split_input_conv.input_name[1])
    now_conv_weight_value = now_conv_weight_node.const_value
    new_shape = now_conv_weight_value.shape[:] + (1,)
    new_now_conv_weight_value = now_conv_weight_value.reshape(new_shape)
    now_conv_weight_node.set_const_value(new_now_conv_weight_value)
    # change the conv_attr
    new_dilations = now_Split_input_conv.get_attr("dilations", AT.LIST_INT) * 2
    new_kernel_shape = now_Split_input_conv.get_attr("kernel_shape", AT.LIST_INT) * 2
    new_pads = now_Split_input_conv.get_attr("pads", AT.LIST_INT) * 2
    new_strides = now_Split_input_conv.get_attr("strides", AT.LIST_INT) * 2
    now_Split_input_conv.set_attr({"dilations": (AT.LIST_INT, new_dilations)})
    now_Split_input_conv.set_attr({"kernel_shape": (AT.LIST_INT, new_kernel_shape)})
    now_Split_input_conv.set_attr({"pads": (AT.LIST_INT, new_pads)})
    now_Split_input_conv.set_attr({"strides": (AT.LIST_INT, new_strides)})

    # change the second conv
    now_split_after_node = mod.get_node(io_map.get(now_Split.name)[1])
    second_conv_node = mod.get_node(io_map.get(now_split_after_node.name)[0])
    now_second_conv_weight_node = mod.get_node(second_conv_node.input_name[1])
    now_second_conv_weight_value = now_second_conv_weight_node.const_value
    new_second_shape = now_second_conv_weight_value.shape[:2] + (1,) + now_second_conv_weight_value.shape[2:3]
    new_now_second_conv_weight_value = now_second_conv_weight_value.reshape(new_second_shape)
    now_second_conv_weight_node.set_const_value(new_now_second_conv_weight_value)
    # change the second conv attr
    new_dilations = second_conv_node.get_attr("dilations", AT.LIST_INT) * 2
    new_kernel_shape = [1] + second_conv_node.get_attr("kernel_shape", AT.LIST_INT)
    new_pads = (
        0, second_conv_node.get_attr("pads", AT.LIST_INT)[0], 0, second_conv_node.get_attr("pads", AT.LIST_INT)[1])
    new_strides = second_conv_node.get_attr("strides", AT.LIST_INT) * 2
    second_conv_node.set_attr({"dilations": (AT.LIST_INT, new_dilations)})
    second_conv_node.set_attr({"kernel_shape": (AT.LIST_INT, new_kernel_shape)})
    second_conv_node.set_attr({"pads": (AT.LIST_INT, new_pads)})
    second_conv_node.set_attr({"strides": (AT.LIST_INT, new_strides)})
    # add the squeeze node to change the shape 1,256,1,64 to 1,256,64

    squeeze_node = mod.add_new_node(now_Split_input_conv.name + "squeeze", "Squeeze",
                                    {"axes": (AT.LIST_INT, [2])
                                     })
    now_mul_after_second_conv = mod.get_node(io_map.get(second_conv_node.name)[1])
    third_conv = mod.get_node(io_map.get(now_mul_after_second_conv.name)[0])
    third_conv.set_input_node(0, [squeeze_node])
    squeeze_node.set_input_node(0, [now_mul_after_second_conv])

# revise the repeat linear_q.bias name
Add_lists = mod.get_nodes_by_optype("Add")
for add_node in Add_lists:
    now_Add = mod.get_node(add_node)
    Add_input2 = mod.get_node(now_Add.input_name[1])

    if Add_input2.op_type in ("Initializer", "Constant") and operator.eq(Add_input2.const_value.shape, (4, 64)):
        Add_input2_val = Add_input2.const_value.flatten()
        Add_input2_before_reshape = mod.get_node(now_Add.input_name[0])
        reshape_before_add = mod.get_node(Add_input2_before_reshape.input_name[0])
        # get the reshape_before_add const and add Add_input2_val
        reshape_before_add_input1 = mod.get_node(reshape_before_add.input_name[0])
        reshape_before_add_val = reshape_before_add_input1.const_value
        reshape_before_add_input2 = mod.get_node(reshape_before_add.input_name[1])
        reshape_before_add_val = reshape_before_add_val.astype("float64")
        Add_input2_val = Add_input2_val.astype("float64")
        add_val_sum = reshape_before_add_val + Add_input2_val
        div_add_val_sum = add_val_sum
        add_val_sum = add_val_sum.astype("float32")
        if "pos_bias_u" in Add_input2.name:
            new_reshape_before_add_input1 = mod.add_const_node(
                reshape_before_add_input1.name + reshape_before_add_input2.name, add_val_sum)
            reshape_before_add.set_input_node(0, [new_reshape_before_add_input1, reshape_before_add_input2])
        if "pos_bias_v" in Add_input2.name:
            div_add_val_sum = div_add_val_sum.astype("float32")
            reshape_before_now_add = mod.get_node(now_Add.input_name[0])
            Add_before_now_reshape = mod.get_node(reshape_before_now_add.input_name[0])
            MatMul_before_now_reshape = mod.get_node(Add_before_now_reshape.input_name[1])
            MatMul_before_now_reshape_input2 = mod.get_node(MatMul_before_now_reshape.input_name[1])
            MatMul_val = MatMul_before_now_reshape_input2.const_value.astype("float64")
            new_MatMul_val = MatMul_val
            new_MatMul_val = new_MatMul_val.astype("float32")
            MatMul_before_now_reshape_input2.set_const_value(new_MatMul_val)
            new_reshape_before_add_input1 = mod.add_const_node(
                reshape_before_add_input1.name + reshape_before_add_input2.name, div_add_val_sum)
            reshape_before_add.set_input_node(0, [new_reshape_before_add_input1, reshape_before_add_input2])
        node_after_now_add = mod.get_node(io_map.get(now_Add.name)[0])
        remove_one_in_one_out_node(mod, [now_Add], Add_input2_before_reshape, node_after_now_add)
    else:
        continue

# move the Add to the before add
# Add1-> Reshape -> Add2  ---->  Add3-> Reshape
Add_lists = mod.get_nodes_by_optype("Add")
for add_node2 in Add_lists:
    now_Add = mod.get_node(add_node2)
    Add_input2 = mod.get_node(now_Add.input_name[1])
    if Add_input2.op_type in ("Initializer", "Constant") and operator.eq(Add_input2.const_value.shape, (4, 64)):
        Add_input2_val = Add_input2.const_value.flatten()
        Add_input2_before_reshape = mod.get_node(now_Add.input_name[0])
        reshape_before_add = mod.get_node(Add_input2_before_reshape.input_name[0])
        # get the reshape_before_add const and add Add_input2_val
        reshape_before_add_input1 = mod.get_node(reshape_before_add.input_name[0])
        reshape_before_add_val = reshape_before_add_input1.const_value
        add_val_sum = reshape_before_add_val + Add_input2_val
        reshape_before_add_input1.set_const_value(add_val_sum)
        node_after_now_add = mod.get_node(io_map.get(now_Add.name)[0])
        remove_one_in_one_out_node(mod, [now_Add], Add_input2_before_reshape, node_after_now_add)
    else:
        continue

mod.save_new_model("encoder_revise.onnx")
