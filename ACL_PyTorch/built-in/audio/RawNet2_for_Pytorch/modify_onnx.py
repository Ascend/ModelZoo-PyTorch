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

# -*- coding:utf-8 -*-

import argparse
import numpy as np
import copy
from gener_core.mod_modify.interface import AttrType as AT
from gener_core.mod_modify.onnx_graph import OXGraph


def make_conv2d_split_node(mod, conv_node, weight, chk_idx, chk_sz, ksz):
    x_node = mod.get_node(conv_node.input_name[0])

    # slice
    rng = np.random.randint(0, 7393)
    begin = chk_sz * chk_idx
    end = -(ksz - chk_sz) + chk_sz * chk_idx
    if end >= 0:
        end = np.iinfo(np.int32).max
    begin1_node = mod.add_const_node(f"const_begin_{rng}", np.array([begin], np.int32))
    end1_node = mod.add_const_node(f"const_end_{rng}", np.array([end], np.int32))
    axes_node = mod.add_const_node(f"const_axes_{rng}", np.array([-1], np.int32))
    step_node = mod.add_const_node(f"const_step_{rng}", np.array([1], np.int32))
    slice1_node = mod.add_new_node(f"Slice_{rng}", "Slice")
    slice1_node.set_input_node(0, [x_node, begin1_node, end1_node, axes_node, step_node])

    # conv
    conv1_node = mod.add_new_node(f"Conv_{np.random.randint(0, 7393)}", "Conv",
                                  {"dilations": (AT.LIST_INT, [1, 1]),
                                   "group": (AT.INT, 1),
                                   "kernel_shape": (AT.LIST_INT, [1, weight.shape[-1]]),
                                   "pads": (AT.LIST_INT, [0, 0, 0, 0]),
                                   "strides": (AT.LIST_INT, [1, 1]), })
    w1_node = mod.add_const_node(f"weight_{np.random.randint(0, 7393)}", weight)
    conv1_node.set_input_node(0, [slice1_node, w1_node])

    return conv1_node


def shape_dim_extend(mod, io_map):
    # NCD -> NCHW
    rshp_node = mod.get_node("Reshape_9")
    shape_node = mod.get_node(rshp_node.input_name[1])
    shape_value = shape_node.const_value
    rng = np.random.randint(0, 7393)
    shape_value = np.insert(shape_value, 2, 1)
    new_shape_node = mod.add_const_node(f"const_shape_{rng}", shape_value.astype(np.int32))
    mod.node_replace(shape_node, new_shape_node)

    # modify all nodes for conv and maxpool
    g_nodes = mod.get_nodes_by_optype("Conv")
    for g_node in g_nodes:
        weight_node = mod.get_node(g_node.input_name[1])
        weight_value = weight_node.const_value

        if len(weight_value.shape) == 3:
            rng = np.random.randint(0, 7393)
            kernel_shape = [1] + g_node.get_attr('kernel_shape', AT.LIST_INT)
            dilations = g_node.get_attr('dilations', AT.LIST_INT) * 2
            pads = g_node.get_attr('pads', AT.LIST_INT)
            if pads == [0, 0]:
                pads = [0, 0, 0, 0]
            if pads == [1, 1]:
                pads = [0, 1, 0, 1]
            strides = g_node.get_attr('strides', AT.LIST_INT) * 2
            g_node.set_attr({"kernel_shape": (AT.LIST_INT, kernel_shape)})
            g_node.set_attr({"dilations": (AT.LIST_INT, dilations)})
            g_node.set_attr({"pads": (AT.LIST_INT, pads)})
            g_node.set_attr({"strides": (AT.LIST_INT, strides)})
            new_weight_node = mod.add_const_node(f"const_weight_{rng}", np.expand_dims(weight_value, axis=2))
            mod.node_replace(weight_node, new_weight_node)

    g_node = mod.get_node("MaxPool_13")
    rng = np.random.randint(0, 7393)
    kernel_shape = [1] + g_node.get_attr('kernel_shape', AT.LIST_INT)
    pads = g_node.get_attr('pads', AT.LIST_INT) * 2
    strides = g_node.get_attr('strides', AT.LIST_INT) * 2
    g_node.set_attr({"kernel_shape": (AT.LIST_INT, kernel_shape),
                     "dilations": (AT.LIST_INT, dilations),
                     "pads": (AT.LIST_INT, pads),
                     "strides": (AT.LIST_INT, strides)})

    # NCHW -> NCD
    res_node = mod.get_node('MaxPool_13')
    squeeze_node = mod.add_new_node(f"Squeeze_{np.random.randint(0, 7393)}", "Squeeze",
                                    {"axes": (AT.LIST_INT, [2])})
    squeeze_node.set_input_node(0, [res_node])
    after_res_node = mod.get_node(io_map.get(res_node.name)[0])
    after_res_node.set_input_node(0, [squeeze_node])

    # NCD -> NCHW
    g_nodes = mod.get_nodes_by_optype("Conv")
    for g_node in g_nodes:
        if g_node.name != "Conv_11" and mod.get_node(g_node.input_name[0]).op_type != "LeakyRelu":
            rng = np.random.randint(0, 7393)
            unsqueeze_node = mod.add_new_node(f"Unsqueeze_{rng}", "Unsqueeze",
                                              {"axes": (AT.LIST_INT, [2])})

            before_g_node = mod.get_node(g_node.input_name[0])
            w_node = mod.get_node(g_node.input_name[1])
            if len(g_node.input_name) == 2:
                g_node.set_input_node(0, [unsqueeze_node, w_node])
            else:
                b_node = mod.get_node(g_node.input_name[2])
                g_node.set_input_node(0, [unsqueeze_node, w_node, b_node])
            unsqueeze_node.set_input_node(0, [before_g_node])

            # NCHW -> NCD
    g_nodes = mod.get_nodes_by_optype("Add")
    for g_node in g_nodes:
        Add_b0 = mod.get_node(g_node.input_name[0])
        Add_b1 = mod.get_node(g_node.input_name[1])
        if mod.get_node(Add_b0.input_name[0]).op_type == "LeakyRelu":
            rng = np.random.randint(0, 7393)
            if Add_b1.op_type != "Conv":
                unsqueeze_node = mod.add_new_node(f"Unsqueeze_{rng}", "Unsqueeze",
                                                  {"axes": (AT.LIST_INT, [2])})
                g_node.set_input_node(0, [unsqueeze_node, Add_b0])
                unsqueeze_node.set_input_node(0, [Add_b1])

            squeeze_node = mod.add_new_node(f"Squeeze_{rng}", "Squeeze",
                                            {"axes": (AT.LIST_INT, [2])})
            squeeze_node.set_input_node(0, [g_node])
            after_g_node = mod.get_node(io_map.get(g_node.name)[0])
            after_g_node.set_input_node(0, [squeeze_node])


def make_model(input_onnx, output_onnx):
    mod = OXGraph(input_onnx)
    io_map = mod.get_net_in_out_map()

    # solve accuracy problem
    gather_nodes = mod.get_nodes_by_optype("Gather")
    for g_node in gather_nodes:
        if g_node.name == 'Gather_203':
            indices_node = mod.add_const_node(f'Const_{g_node.input_name[1]}', np.array(28).astype('int64'))
            g_node.set_input_node(1, [indices_node])

    # NCD -> NCHW
    shape_dim_extend(mod, io_map)

    # conv split
    conv_node = mod.get_node("Conv_11")
    weight_node = mod.get_node(conv_node.input_name[1])
    weight_value = weight_node.const_value

    np.random.seed(1737)
    KSZ = weight_value.shape[-1]
    CHK_SZ = 128
    CHK_N = KSZ // CHK_SZ
    wgt = []
    for i in range(CHK_N):
        wgt.append(weight_value[:, :, :, CHK_SZ * i:CHK_SZ * (i + 1)])
    if KSZ % CHK_SZ != 0:
        wgt.append(weight_value[:, :, :, CHK_SZ * CHK_N:])

    rwn_node = []
    for i, w in enumerate(wgt):
        node = make_conv2d_split_node(mod, conv_node, w, i, CHK_SZ, KSZ)
        rwn_node.append(node)

    in_node_list = copy.deepcopy(rwn_node[:CHK_N])
    out_node_list = []
    combin_len = CHK_N
    while len(in_node_list) > 1:
        for j in range(0, combin_len, 2):
            add_node = mod.add_new_node(f"Add_{np.random.randint(0, 7393)}", "Add")
            add_node.set_input_node(0, [in_node_list[j], in_node_list[j + 1]])
            out_node_list.append(add_node)
        in_node_list = copy.deepcopy(out_node_list)
        out_node_list.clear()
        combin_len //= 2

    # add all result
    if KSZ % CHK_SZ != 0:
        add_node = mod.add_new_node(f"Add_{np.random.randint(0, 7393)}", "Add")
        add_node.set_input_node(0, [in_node_list[0], rwn_node[-1]])
    else:
        add_node = in_node_list[0]

    # relink
    after_node = mod.get_node(io_map.get(conv_node.name)[0])
    after_node.set_input_node(0, [add_node])

    # remove ori node
    mod.node_remove([conv_node.name])
    mod.save_new_model(output_onnx)


def get_parser():
    parser = argparse.ArgumentParser(description='RawNet2')
    parser.add_argument('--input_onnx', default=None, type=str,
                        help='input original onnx')
    parser.add_argument('--output_onnx', default=None, type=str,
                        help='output modified onnx')
    return parser


if __name__ == "__main__":
    '''
    Example:
        python3.7 modify_onnx.py \
            --input_onnx=rawnet2_sim.onnx \
            --output_onnx=rawnet2_modify.onnx
    '''
    parser = get_parser()
    args = parser.parse_args()
    make_model(args.input_onnx, args.output_onnx)
    print("modify successfully!")
