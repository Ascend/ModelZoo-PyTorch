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


##################
# basic function #
##################
def insert_shape_node(anchor_name, dst_shape, mode='after', index=0):
    reshape_node = onnx_graph.add_node(
        f"Reshape_{mode}_{anchor_name}",
        "Reshape"
    )
    onnx_graph.insert_node(
        anchor_name,
        reshape_node,
        refer_index=index,
        mode=mode
    )
    reshape_init = onnx_graph.add_initializer(
        f"{reshape_node.name}_shape",
        np.array(dst_shape, dtype="int64")
    )
    reshape_node.inputs.append(reshape_init.name)


def insert_slice_node(anchor_name, start_value, end_value, axis_value, mode='after', index=0):
    slice_node = onnx_graph.add_node(
        f"Slice_{mode}_{anchor_name}",
        "Slice"
    )
    onnx_graph.insert_node(anchor_name, slice_node,
                           refer_index=index, mode=mode)
    slice_start = onnx_graph.add_initializer(
        f"start_{slice_node.name}",
        np.array(start_value, dtype="int64")
    )
    slice_end = onnx_graph.add_initializer(
        f"end_{slice_node.name}",
        np.array(end_value, dtype="int64")
    )
    slice_axis = onnx_graph.add_initializer(
        f"axis_{slice_node.name}",
        np.array(axis_value, dtype="int64")
    )
    slice_node.inputs.append(slice_start.name)
    slice_node.inputs.append(slice_end.name)
    slice_node.inputs.append(slice_axis.name)


def insert_concat_node(anchor_name, concat_value, axis=0, mode='after'):
    concat_node = onnx_graph.add_node(
        f"Concat_{mode}_{anchor_name}",
        "Concat",
        attrs={
            "axis": axis
        }
    )
    onnx_graph.insert_node(anchor_name, concat_node, mode=mode)
    concat_init = onnx_graph.add_initializer(
        f"init_{concat_node.name}",
        concat_value
    )
    concat_node.inputs.append(concat_init.name)


#################
# core function #
#################
def merge_bmm_axis(onnx_graph, first_add_name, reshape_nodes, last_add_node, bs, hidden_dim):
    # merge 3 dim input to 2 dim: improve performance for bmm
    insert_shape_node(first_add_name, [-1, hidden_dim], mode='before')

    # merge const value for first add node
    first_add_node = onnx_graph[first_add_name]
    first_add_value = onnx_graph[first_add_node.inputs[1]].value
    onnx_graph[first_add_node.inputs[1]].value = np.tile(
        first_add_value, [bs, 1, 1]).reshape(-1, hidden_dim)

    # modify dst shape value dim: 3 -> 2
    for reshape_name in reshape_nodes:
        reshape_node = onnx_graph[reshape_name]
        reshape_init = onnx_graph[onnx_graph[reshape_name].inputs[1]]
        reshape_init.value = np.array([-1, hidden_dim], dtype="int64")

    # rescover model output dim: 2 -> 3
    insert_shape_node(last_add_node, [bs, -1, hidden_dim])


def pad_nz_block(onnx_graph, first_add_node, reshape_nodes,
                 add_nodes, last_add_node, bs, hidden_dim1,
                 hidden_dim2, padding_size):
    # pad input to 16 aligned with zero value: friendly to hardware
    first_concat_value = np.zeros([padding_size * bs, hidden_dim2]).astype("float32")
    insert_concat_node(first_add_node, first_concat_value, mode="before")
    add_init = onnx_graph[onnx_graph[first_add_node].inputs[1]]
    assert add_init.op_type == "Initializer", "second input for first add must be initializer!"
    add_init.value = np.concatenate(
        [add_init.value, np.zeros([padding_size * bs, hidden_dim2])],
        axis=0).astype("float32")

    # unpad before attention block
    for reshape_name in reshape_nodes:
        insert_slice_node(reshape_name, [0], [hidden_dim1 * bs], [0], mode="before")

    # pad again after attention block
    concat_value = np.zeros([padding_size * bs, hidden_dim2]).astype("float32")
    for add_name in add_nodes:
        insert_concat_node(add_name, concat_value)

    # unpad to rescover output ori shape
    insert_slice_node(last_add_node, [0], [hidden_dim1 * bs], [0])


def get_attention_reshape_nodes(onnx_graph, dst_shape,
                                check_pre_nodes=None, check_next_nodes=None,
                                pre_nodes_idxes=None, next_nodes_idxes=None):
    node_list = []
    for reshape_node in onnx_graph.get_nodes(op_type="Reshape"):
        shape_input0 = onnx_graph.get_prev_node(reshape_node.inputs[0])
        shape_input1 = onnx_graph.get_prev_node(reshape_node.inputs[1])
        if shape_input0 is None:
            out_shape = onnx_graph[reshape_node.inputs[0]]
        elif shape_input1 is None:
            out_shape = onnx_graph[reshape_node.inputs[1]]
        else:
            continue

        if out_shape.value.shape != dst_shape:
            continue
        check_succed = True
        if check_pre_nodes is not None:
            cur_node = reshape_node
            for idx, pre_node_type in enumerate(check_pre_nodes):
                cur_node = onnx_graph.get_prev_node(cur_node.inputs[pre_nodes_idxes[idx]])
                if cur_node is None or cur_node.op_type != pre_node_type:
                    check_succed = False
                    break
        if check_next_nodes is not None:
            cur_node = reshape_node
            for idx, next_node_type in enumerate(check_next_nodes):
                cur_node = onnx_graph.get_next_nodes(cur_node.outputs[0])[next_nodes_idxes[idx]]
                if cur_node is None or cur_node.op_type != next_node_type:
                    check_succed = False
                    break
        if check_succed:
            node_list.append(reshape_node.name)
    return node_list


def get_attention_add_nodes(onnx_graph, weight_shape,
                            check_pre_nodes=None, check_next_nodes=None,
                            pre_nodes_idxes=None, next_nodes_idxes=None):
    node_list = []
    for add_node in onnx_graph.get_nodes(op_type="Add"):
        add_input0 = onnx_graph.get_prev_node(add_node.inputs[0])
        add_input1 = onnx_graph.get_prev_node(add_node.inputs[1])
        if add_input0 is None:
            add_weight = onnx_graph[add_node.inputs[0]]
        elif add_input1 is None:
            add_weight = onnx_graph[add_node.inputs[1]]
        else:
            continue

        if add_weight.value.shape != weight_shape:
            continue
        check_succed = True
        if check_pre_nodes is not None:
            cur_node = add_node
            for idx, pre_node_type in enumerate(check_pre_nodes):
                cur_node = onnx_graph.get_prev_node(cur_node.inputs[pre_nodes_idxes[idx]])
                if cur_node is None or cur_node.op_type != pre_node_type:
                    check_succed = False
                    break
        if check_next_nodes is not None:
            cur_node = add_node
            for idx, next_node_type in enumerate(check_next_nodes):
                cur_node = onnx_graph.get_next_nodes(cur_node.outputs[0])[next_nodes_idxes[idx]]
                if cur_node is None or cur_node.op_type != next_node_type:
                    check_succed = False
                    break
        if check_succed:
            node_list.append(add_node.name)
    return node_list


def cal_model_para(onnx_graph):
    input_shape = onnx_graph.inputs[0].shape
    assert input_shape[-1] == input_shape[-2], "input h != w"
    input_size = input_shape[-1]
    conv_node = onnx_graph.get_nodes(op_type="Conv")[0]
    assert conv_node["kernel_shape"][0] == conv_node["kernel_shape"][1]
    conv_kernel = conv_node["kernel_shape"][0]
    assert conv_node["pads"][0] == conv_node["pads"][1]
    assert conv_node["pads"][1] == conv_node["pads"][2]
    assert conv_node["pads"][2] == conv_node["pads"][3]
    conv_pad = conv_node["pads"][0]
    assert conv_node["strides"][0] == conv_node["strides"][1]
    conv_stride = conv_node["strides"][0]
    output_size = (input_size - conv_kernel + 2 * conv_pad) / conv_stride + 1
    hidden_dim = int(output_size * output_size + 1)
    if hidden_dim % 16 == 0:
        padding_size = 0
    else:
        padding_size = int((hidden_dim // 16 + 1) * 16 - hidden_dim)
    return hidden_dim, padding_size


if __name__ == '__main__':
    onnx_graph = OnnxGraph.parse(sys.argv[1])
    bs = int(sys.argv[3])
    hidden_dim2 = 768
    num_block = 12

    # cal hidden_dim1 && padding size
    hidden_dim1, padding_size = cal_model_para(onnx_graph)
    reshape_nodes = get_attention_reshape_nodes(onnx_graph, dst_shape=(3,),
                                                check_pre_nodes=["Transpose", "MatMul"],
                                                pre_nodes_idxes=[0, 0])
    add_nodes = onnx_graph.get_nodes(op_type="Add")
    add_nodes = sorted(add_nodes, key=lambda x:int(x.name.split("_")[1]))
    # merge input tensor for bmm
    merge_bmm_axis(onnx_graph,
                   first_add_name=add_nodes[0].name,
                   reshape_nodes=reshape_nodes,
                   last_add_node=add_nodes[-1].name,
                   bs=bs,
                   hidden_dim=hidden_dim2)

    reshape_nodes = get_attention_reshape_nodes(onnx_graph, dst_shape=(5,),
                                                check_next_nodes=["Transpose", "Split"],
                                                next_nodes_idxes=[0, 0])
    assert len(reshape_nodes) == num_block, "num of reshape node must equal to num of block."
    att_add_nodes = get_attention_add_nodes(onnx_graph, weight_shape=(hidden_dim2,),
                                            check_pre_nodes=["MatMul", "Reshape"],
                                            pre_nodes_idxes=[1, 0]
                                            )
    if not att_add_nodes:
        att_add_nodes = get_attention_add_nodes(onnx_graph, weight_shape=(hidden_dim2,),
                                                check_pre_nodes=["MatMul", "Reshape"],
                                                pre_nodes_idxes=[0, 0]
                                                )
    assert len(att_add_nodes) == num_block, "num of add node must equal to num of block."
    # pad input shape for attention block: keep 16 aligned
    pad_nz_block(onnx_graph,
                 first_add_node=add_nodes[0].name,
                 reshape_nodes=reshape_nodes,
                 add_nodes=att_add_nodes,
                 last_add_node=add_nodes[-1].name,
                 bs=bs,
                 hidden_dim1=hidden_dim1,
                 hidden_dim2=hidden_dim2,
                 padding_size=padding_size)

    # check infer shape
    onnx_graph.infershape()

    onnx_graph.save(sys.argv[2])
