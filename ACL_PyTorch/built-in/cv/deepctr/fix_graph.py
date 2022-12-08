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
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph


def change_case(g):
    cast_list = g.get_nodes('Cast')
    for cast in cast_list:
        g.remove(cast.name)

    new_node = g.add_node('new_cast', 'Cast', attrs={'to': 6})
    g.connect_node(
        new_node,
        ['input'],
        [g.get_nodes('Slice')[0].name]
    )
    return g


def fix_right_wdl(g):
    gather_list = g.get_nodes('Gather')
    slice_list = g.get_nodes('Slice')
    # get the input of gather, data_ is the merged data
    data_list = []
    for gather in gather_list:
        g_name = gather.name
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx < 48:
            data_input = gather.inputs[0]
            data_list.append(g[data_input].value)
    # merge data
    data_ = data_list[0]
    for i in range(1, 6):
        data_ = np.concatenate([data_, data_list[i]], axis=0)

    # del slice
    for slice in slice_list:
        s_name = slice.name
        s_name_idx = int(s_name.split('_')[1])
        if s_name_idx < 46:
            g.remove(s_name, {})

    # add Add
    new_add_right = g.add_node('new_add_right', 'Add')
    list_ind = []
    for j in range(40):
        temp = []
        for i in [0, 187, 380, 382, 389, 409]:
            temp.append(i)
        list_ind.append(temp)
    add_other_right = g.add_initializer('add_other_right',
                                        np.array(list_ind, dtype=np.int32))
    g.connect_node(
        new_add_right,
        ['new_cast', 'add_other_right'],
        ['Flatten_90']
    )

    # fix gather
    for gather in gather_list:
        g_name = gather.name
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx < 48:
            g.remove(g_name, {})

    data_right = g.add_initializer('data_right', data_)
    new_gather_right = g.add_node('new_gather_right', 'Gather')
    g.connect_node(
        new_gather_right,
        ['data_right', 'new_add_right'],
        ['Flatten_90']
    )
    return g


def fix_left_wdl(g):
    gather_list = g.get_nodes('Gather')
    slice_list = g.get_nodes('Slice')
    # get the input of gather, data_ is the merged data
    data_list = []
    for gather in gather_list:
        g_name = gather.name
        if g_name == 'new_gather_right':
            continue
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx >= 48:
            data_input = gather.inputs[0]
            data_list.append(g[data_input].value)
    # merge data
    data_ = data_list[0]
    for i in range(1, 6):
        data_ = np.concatenate([data_, data_list[i]], axis=0)

    # del slice
    for slice in slice_list:
        s_name = slice.name
        s_name_idx = int(s_name.split('_')[1])
        if s_name_idx >= 46:
            g.remove(s_name, {})

    # fix gather
    for gather in gather_list:
        g_name = gather.name
        if g_name == 'new_gather_right':
            continue
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx >= 48:
            g.remove(g_name, {})

    data_left = g.add_initializer('data_left', data_)
    new_gather_left = g.add_node('new_gather_left', 'Gather')
    g.connect_node(
        new_gather_left,
        ['data_left', 'new_add_right'],
        ['ReduceSum_86']
    )
    # del concat
    concat_list = g.get_nodes('Concat')
    for concat in concat_list:
        c_name = concat.name
        g.remove(c_name, {})
    # shape inplace concat
    new_reshape = g.add_node('new_shape', 'Reshape')
    shape = g.add_initializer('shape', np.array([40, 1, 6], dtype=np.int64()))
    g.connect_node(
        new_reshape,
        ['new_gather_left', 'shape'],
        ['ReduceSum_86']
    )
    return g


def fix_right_AutoInt(g):
    gather_list = g.get_nodes('Gather')
    slice_list = g.get_nodes('Slice')
    # get the input of gather, data_ is the merged data
    data_list = []
    for gather in gather_list:
        g_name = gather.name
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx < 48:
            data_input = gather.inputs[0]
            data_list.append(g[data_input].value)
    # merge data
    data_ = data_list[0]
    for i in range(1, 6):
        data_ = np.concatenate([data_, data_list[i]], axis=0)

    # del slice
    for slice in slice_list:
        s_name = slice.name
        s_name_idx = int(s_name.split('_')[1])
        if s_name_idx < 46:
            g.remove(s_name, {})

    # add Add
    new_add_right = g.add_node('new_add_right', 'Add')
    list_ind = []
    for j in range(40):
        temp = []
        for i in [0, 187, 380, 382, 389, 409]:
            temp.append(i)
        list_ind.append(temp)
    add_other_right = g.add_initializer('add_other_right', np.array(list_ind, dtype=np.int32))
    g.connect_node(
        new_add_right,
        ['new_cast', 'add_other_right'],
        ['Flatten_434']
    )

    # fix gather
    for gather in gather_list:
        g_name = gather.name
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx < 48:
            g.remove(g_name, {})

    data_right = g.add_initializer('data_right', data_)
    new_gather_right = g.add_node('new_gather_right', 'Gather')
    g.connect_node(
        new_gather_right,
        ['data_right', 'new_add_right'],
        ['Flatten_434']
    )
    new_reshape_right1 = g.add_node('new_shape_right1', 'Reshape')
    shape_right1 = g.add_initializer('shape_right1', np.array([240, 4], dtype=np.int64()))
    g.connect_node(
        new_reshape_right1,
        ['new_gather_right', 'shape_right1'],
        ['Einsum_199:0;Einsum_144:0;Einsum_106:0;Einsum_125:0']
    )
    for i in [105, 124, 143, 198]:
        shape_name = 'Reshape_' + str(i)
        g.remove(shape_name, {})

    new_reshape_right2 = g.add_node('new_shape_right2', 'Reshape')
    shape_right2 = g.add_initializer('shape_right2', np.array([40, 1, 24], dtype=np.int64()))
    g.connect_node(
        new_reshape_right2,
        ['new_gather_right', 'shape_right2'],
        ['Flatten_434']
    )
    return g


def fix_left_AutoInt(g):
    gather_list = g.get_nodes('Gather')
    slice_list = g.get_nodes('Slice')
    # get the input of gather, data_ is the merged data
    data_list = []
    for gather in gather_list:
        g_name = gather.name
        if g_name == 'new_gather_right':
            continue
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx >= 48:
            data_input = gather.inputs[0]
            data_list.append(g[data_input].value)
    # merge data
    data_ = data_list[0]
    for i in range(1, 6):
        data_ = np.concatenate([data_, data_list[i]], axis=0)

    # del slice
    for slice in slice_list:
        s_name = slice.name
        s_name_idx = int(s_name.split('_')[1])
        if s_name_idx >= 46:
            g.remove(s_name, {})

    # fix gather
    for gather in gather_list:
        g_name = gather.name
        if g_name == 'new_gather_right':
            continue
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx >= 48:
            g.remove(g_name, {})

    data_left = g.add_initializer('data_left', data_)
    new_gather_left = g.add_node('new_gather_left', 'Gather')
    g.connect_node(
        new_gather_left,
        ['data_left', 'new_add_right'],
        ['ReduceSum_86']
    )

    # shape inplace concat
    new_reshape_left = g.add_node('new_shape_left', 'Reshape')
    shape_left = g.add_initializer('shape_left', np.array([40, 1, 6], dtype=np.int64()))
    g.connect_node(
        new_reshape_left,
        ['new_gather_left', 'shape_left'],
        ['ReduceSum_86']
    )
    for i in [84, 89, 433]:
        c_name = 'Concat_' + str(i)
        g.remove(c_name, {})
    return g


def fix_right_xDeepFM(g):
    gather_list = g.get_nodes('Gather')
    slice_list = g.get_nodes('Slice')
    # get the input of gather, data_ is the merged data
    data_list = []
    for gather in gather_list:
        g_name = gather.name
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx >= 48:
            data_input = gather.inputs[0]
            data_list.append(g[data_input].value)
    # merge data
    data_ = data_list[0]
    for i in range(1, 6):
        data_ = np.concatenate([data_, data_list[i]], axis=0)

    # del slice
    for slice in slice_list:
        s_name = slice.name
        s_name_idx = int(s_name.split('_')[1])
        if s_name_idx >= 46:
            g.remove(s_name, {})

    # add Add
    new_add_right = g.add_node('new_add_right', 'Add')
    list_ind = []
    for j in range(40):
        temp = []
        for i in [0, 187, 380, 382, 389, 409]:
            temp.append(i)
        list_ind.append(temp)
    add_other_right = g.add_initializer('add_other_right', np.array(list_ind, dtype=np.int32))
    g.connect_node(
        new_add_right,
        ['new_cast', 'add_other_right'],
        ['ReduceSum_86']
    )

    # fix gather
    for gather in gather_list:
        g_name = gather.name
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx >= 48:
            g.remove(g_name, {})

    data_right = g.add_initializer('data_right', data_)
    new_gather_right = g.add_node('new_gather_right', 'Gather')
    g.connect_node(
        new_gather_right,
        ['data_right', 'new_add_right'],
        ['ReduceSum_86']
    )
    new_reshape_right = g.add_node('new_shape_right', 'Reshape')
    shape_right = g.add_initializer('shape_right', np.array([40, 1, 6], dtype=np.int64()))
    g.connect_node(
        new_reshape_right,
        ['new_gather_right', 'shape_right'],
        ['ReduceSum_86']
    )

    return g


def fix_left_xDeepFM(g):
    gather_list = g.get_nodes('Gather')
    slice_list = g.get_nodes('Slice')
    # get the input of gather, data_ is the merged data
    data_list = []
    for gather in gather_list:
        g_name = gather.name
        if g_name == 'new_gather_right':
            continue
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx < 48:
            data_input = gather.inputs[0]
            data_list.append(g[data_input].value)
    # merge data
    data_ = data_list[0]
    for i in range(1, 6):
        data_ = np.concatenate([data_, data_list[i]], axis=0)

    # del slice
    for slice in slice_list:
        s_name = slice.name
        s_name_idx = int(s_name.split('_')[1])
        if s_name_idx < 46:
            g.remove(s_name, {})

    # fix gather
    for gather in gather_list:
        g_name = gather.name
        if g_name == 'new_gather_right':
            continue
        g_name_idx = int(g_name.split('_')[1])
        if g_name_idx < 48:
            g.remove(g_name, {})

    data_left = g.add_initializer('data_left', data_)
    new_gather_left = g.add_node('new_gather_left', 'Gather')
    g.connect_node(
        new_gather_left,
        ['data_left', 'new_add_right'],
        ['Einsum_90:0,1']
    )

    node_n = g['Einsum_96']
    node_n.inputs = ['129', 'new_gather_left_out_0']
    g.update_map()

    new_reshape_left = g.add_node('new_shape_left', 'Reshape')
    shape_left = g.add_initializer('shape_left', np.array([40, 1, 24], dtype=np.int64()))
    g.connect_node(
        new_reshape_left,
        ['new_gather_left', 'shape_left'],
        ['Flatten_105']
    )

    for i in [89, 104, 84]:
        c_name = 'Concat_' + str(i)
        g.remove(c_name, {})
    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', default="./WDL.onnx", type=str, help="dir of input model name")
    parser.add_argument('--output_model', default="./WDL_fix.onnx", type=str, help='dir of output model name')
    args = parser.parse_args()

    # load onnx
    model_path = args.input_model
    out_model_path = args.output_model
    graph = OnnxGraph.parse(model_path)

    model_name = model_path.split('/')[-1].split('_')[0]
    print('change model name is', model_name)
    graph = change_case(graph)
    if model_name == 'WDL':
        graph = fix_right_wdl(graph)
        graph = fix_left_wdl(graph)
    elif model_name == 'AutoInt':
        graph = fix_right_AutoInt(graph)
        graph = fix_left_AutoInt(graph)
    elif model_name == 'xDeepFM':
        graph = fix_right_xDeepFM(graph)
        graph = fix_left_xDeepFM(graph)
    else:
        print('please enter the correct name of model')

    graph.save(out_model_path)