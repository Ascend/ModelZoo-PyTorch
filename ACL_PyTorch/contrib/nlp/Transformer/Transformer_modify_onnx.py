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
import onnx
import argparse
import copy
import numpy as np
from gener_core.mod_modify.onnx_graph import OXGraph


def insert_cast_op(input_model_path, output_model_path):
    '''
    om的ScatterND和Slice算子不支持int64，因此在onnx此算子前后插入Cast算子，实现：int64 -> int32 -> int64
    '''

    model = onnx.load(input_model_path)
    nodes = model.graph.node

    for node in nodes:
        if node.op_type == "ScatterND":
            if node.name == "ScatterND_1321":
                cast_bef = onnx.helper.make_node(
                    'Cast',
                    name='Cast_bef_' + node.name,
                    inputs=[node.input[2]],
                    outputs=[node.input[2] + "_S1321_1"],
                    to=int(onnx.TensorProto.INT32),
                )

                cast_aft = onnx.helper.make_node(
                    'Cast',
                    name='Cast_aft_' + node.name,
                    inputs=[node.output[0] + "_S1321_1"],
                    outputs=[node.output[0]],
                    to=int(onnx.TensorProto.INT64),
                )

                node.input[2] = node.input[2] + "_S1321_1"
                node.output[0] = node.output[0] + "_S1321_1"

                nodes.append(cast_bef)
                nodes.append(cast_aft)
            else:
                suffix_num = node.name.split("_")[1]

                cast_bef1 = onnx.helper.make_node(
                    'Cast',
                    name='Cast_bef1_' + node.name,
                    inputs=[node.input[0]],
                    outputs=[node.input[0] + "_S" + suffix_num + "_1"],
                    to=int(onnx.TensorProto.INT32),
                )

                cast_bef2 = onnx.helper.make_node(
                    'Cast',
                    name='Cast_bef2_' + node.name,
                    inputs=[node.input[2]],
                    outputs=[node.input[2] + "_S" + suffix_num + "_2"],
                    to=int(onnx.TensorProto.INT32),
                )

                cast_aft = onnx.helper.make_node(
                    'Cast',
                    name='Cast_aft_' + node.name,
                    inputs=[node.output[0] + "_S" + suffix_num + "_1"],
                    outputs=[node.output[0]],
                    to=int(onnx.TensorProto.INT64),
                )

                node.input[0] = node.input[0] + "_S" + suffix_num + "_1"
                node.input[2] = node.input[2] + "_S" + suffix_num + "_2"
                node.output[0] = node.output[0] + "_S" + suffix_num + "_1"

                nodes.append(cast_bef1)
                nodes.append(cast_bef2)
                nodes.append(cast_aft)
        elif node.op_type == "Slice":
            suffix_num = node.name.split("_")[1]

            cast_bef = onnx.helper.make_node(
                'Cast',
                name='Cast_bef_' + node.name,
                inputs=[node.input[0]],
                outputs=[node.input[0] + "_S" + suffix_num + "_1"],
                to=int(onnx.TensorProto.INT32),
            )

            cast_aft = onnx.helper.make_node(
                'Cast',
                name='Cast_aft_' + node.name,
                inputs=[node.output[0] + "_S" + suffix_num + "_1"],
                outputs=[node.output[0]],
                to=int(onnx.TensorProto.INT64),
            )

            node.input[0] = node.input[0] + "_S" + suffix_num + "_1"
            node.output[0] = node.output[0] + "_S" + suffix_num + "_1"

            nodes.append(cast_bef)
            nodes.append(cast_aft)

    # 保存临时文件
    temp_model_path = os.path.join(os.path.dirname(output_model_path), "temp.onnx")
    onnx.save(model, temp_model_path)
    return temp_model_path


def modify_initializer_type(temp_model_path, output_model_path):
    '''
    om的ScatterND和Slice算子不支持int64，把这两种算子的Initializer输入由int64转为int32
    '''

    model = OXGraph(temp_model_path)

    # 删除临时文件
    os.remove(temp_model_path)

    nodes = model.get_nodes_by_optype("ScatterND")
    nodes.extend(model.get_nodes_by_optype("Slice"))

    for node in nodes:
        if node.op_type == "ScatterND":
            if node.name == "ScatterND_1321":
                input_ind = node.input_name[0]
                input_ind = model.get_node(input_ind)
                val = input_ind.const_value.astype("int32")
                input_ind.set_const_value(val)
        elif node.op_type == "Slice":
            for i in range(1, 5):
                input_ind = node.input_name[i]
                input_ind = model.get_node(input_ind)
                val = input_ind.const_value.astype("int32")
                input_ind.set_const_value(val)

    # 保存临时文件
    temp_model_path_ = os.path.join(os.path.dirname(output_model_path), "temp.onnx")
    model.save_new_model(temp_model_path_)
    return temp_model_path_


def process_gather_op_indices(temp_model_path, output_model_path):
    '''
    om的GatherV2D算子的indices不支持-1输入，因此需要处理onnx中indices为-1的Gather算子
    '''

    model = onnx.load(temp_model_path)

    # 删除临时文件
    os.remove(temp_model_path)

    graph = model.graph
    initializers = graph.initializer
    value_info = graph.value_info
    nodes = graph.node
    print("old initializers len ", len(initializers))

    initializer_index_dict = {}
    for index, elem in enumerate(initializers):
        initializer_index_dict[elem.name] = index

    value_info_index_dict = {}
    for index, elem in enumerate(value_info):
        value_info_index_dict[elem.name] = index

    count = 0
    modify_initializer_name_value = {}
    for node in nodes:
        if node.op_type == "Gather":
            tensor_name = node.input[0]
            indices_name = node.input[1]
            if indices_name in initializer_index_dict:
                if initializers[
                    initializer_index_dict[indices_name]].raw_data == b"\377\377\377\377\377\377\377\377":  # -1
                    print(node.name, "indices name: " + indices_name + ", type: Initializer, value: -1")
                    copy_initializer = copy.deepcopy(initializers[initializer_index_dict[indices_name]])

                    copy_initializer.name = initializers[initializer_index_dict[indices_name]].name + "_" + str(count)

                    value = value_info[value_info_index_dict[tensor_name]].type.tensor_type.shape.dim[0].dim_value - 1
                    value = np.array(value, dtype=np.int64)

                    node.input[1] = copy_initializer.name
                    initializers.append(copy_initializer)
                    modify_initializer_name_value[copy_initializer.name] = value

                    count += 1

    print("initializer need to modify value:\n", modify_initializer_name_value)
    print("new initializers len ", len(initializers))

    # 保存临时文件
    temp_model_path_ = os.path.join(os.path.dirname(output_model_path), "temp.onnx")
    onnx.save(model, temp_model_path_)

    # 用OXGraph修改initializer的值
    model = OXGraph(temp_model_path_)

    # 删除临时文件
    os.remove(temp_model_path_)

    # 修改值
    for key, value in modify_initializer_name_value.items():
        initializer_node = model.get_node(key)
        initializer_node.set_const_value(value)

    temp_model_path_ = os.path.join(os.path.dirname(output_model_path), "temp.onnx")
    model.save_new_model(temp_model_path_)
    return temp_model_path_


def process_topk_op(temp_model_path, output_model_path):
    '''
    由于TopK和Slice算子公用一个Initializers，前边修改Slice算子时把类型改为了int32，和TopK需求不匹配
    '''
    model = onnx.load(temp_model_path)

    # 删除临时文件
    os.remove(temp_model_path)

    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node
    print("old initializers len ", len(initializers))

    # 构造一个新Initializer给TopK用
    copy_initializer = copy.deepcopy(initializers[0])
    copy_initializer.name = "topk_initializer"
    value = np.array([1], dtype=np.int64)
    initializers.append(copy_initializer)
    print("TopK Initializers:\n", copy_initializer)

    # 修改TopK算子的输入Initializer
    for node in nodes:
        if node.op_type == "TopK":
            print("TopK old input is: ", node.input)
            node.input[1] = copy_initializer.name
            print("TopK new input is: ", node.input)
    print("new initializers len ", len(initializers))

    # 保存临时文件
    temp_model_path_ = os.path.join(os.path.dirname(output_model_path), "temp.onnx")
    onnx.save(model, temp_model_path_)

    # 用OXGraph修改initializer的值
    model = OXGraph(temp_model_path_)

    # 删除临时文件
    os.remove(temp_model_path_)

    # 修改值
    initializer_node = model.get_node(copy_initializer.name)
    initializer_node.set_const_value(value)

    model.save_new_model(output_model_path)


if __name__ == "__main__":
    """
    Usage Example:
    python modify_onnx.py \
    --input_model_path ./model/transformer_greedySearch_input15_maxSeqLen15_sim.onnx \
    --output_model_path ./model/transformer_greedySearch_input15_maxSeqLen15_sim_mod.onnx
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_model_path', required=True)
    parser.add_argument('--output_model_path', required=True)

    opt = parser.parse_args()

    temp_model_path = insert_cast_op(opt.input_model_path, opt.output_model_path)
    temp_model_path = modify_initializer_type(temp_model_path, opt.output_model_path)
    temp_model_path_ = process_gather_op_indices(temp_model_path, opt.output_model_path)
    process_topk_op(temp_model_path, opt.output_model_path)
