# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import numpy as np
from magiconnx import OnnxGraph
import argparse
def conv1d2conv2d(graph, node_conv):
    if node_conv == 'Conv_1':
        graph[node_conv]['dilations'] = [1, 1]
        m = graph[node_conv]['kernel_shape'][0]
        graph[node_conv]['kernel_shape'] = [1, m]
        graph[node_conv]['pads'] = [0, 0, 0, 0]
        graph[node_conv]['strides'] = [1, 5]
        wight = graph[graph[node_conv].node.input[1]].value
        graph[graph[node_conv].node.input[1]].value = np.expand_dims(wight, axis=2)
    else:
        graph[node_conv]['dilations'] = [1, 1]
        m = graph[node_conv]['kernel_shape'][0]
        graph[node_conv]['kernel_shape'] = [1, m]
        graph[node_conv]['pads'] = [0, 0, 0, 0]
        graph[node_conv]['strides'] = [1, 2]
        wight = graph[graph[node_conv].node.input[1]].value
        graph[graph[node_conv].node.input[1]].value = np.expand_dims(wight, axis=2)

def transfer_structure(graph, beg_node, end_node):
    if beg_node != 'Conv_71':
        squeeze_name = end_node + 'squeeze'
        OnnxGraph.add_node(graph, squeeze_name, 'Squeeze', attrs={'axes': [2]})
        graph.insert_node(end_node, graph[squeeze_name], mode='after')
    if beg_node != 'Conv_89':
        unsqueeze_name = beg_node + 'unsqueeze'
        OnnxGraph.add_node(graph, unsqueeze_name, 'Unsqueeze', attrs={'axes': [2]})
        graph.insert_node(beg_node, graph[unsqueeze_name], mode='before')
    conv1d2conv2d(graph, beg_node)

def fix_conv1d(model_path, out_path, beg_list, end_list, transpose1_list, transpose2_list):
    graph = OnnxGraph(model_path)
    change_perm(graph, transpose1_list, transpose2_list)
    for idx, beg_node in enumerate(beg_list):
        end_node = end_list[idx]
        transfer_structure(graph, beg_node, end_node)
    graph.save(out_path)

def change_reduce(graph, reduce_list):
    for name in reduce_list:
        graph[name]['axes'] = 1

def change_weight(graph, transpose2_list, mul_list, add_list):
    for idx, name in enumerate(mul_list):
        graph[graph[name].node.input[1]].value = graph[graph[name].node.input[1]].value.reshape(-1, 1, 1)
        graph[graph[add_list[idx]].node.input[1]].value = \
            graph[graph[add_list[idx]].node.input[1]].value.reshape(-1, 1, 1)
        graph[add_list[idx]].node.output[0] = graph[transpose2_list[idx]].node.output[0]
        graph.del_node(transpose2_list[idx], auto_connection=False)

def del_trans(graph, transpose1_list, beg_list):
    for idx, name in enumerate(transpose1_list):
        graph[beg_list[idx]].node.output[0] = graph[name].node.output[0]
        graph.del_node(name, auto_connection=False)

def change_perm(graph, transpose1_list, transpose2_list):
    for trans1 in transpose1_list:
        graph[trans1]['perm'] = [0, 2, 3, 1]
    for trans2 in transpose2_list:
        graph[trans2]['perm'] = [0, 3, 1, 2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='ground truth text', default="data2vec.onnx")
    parser.add_argument('--output_path', help='infered text', default="data2vec_new.onnx")
    args = parser.parse_args()
    beg_node = ['Conv_1', 'Conv_24', 'Conv_46', 'Conv_68', 'Conv_90', 'Conv_112', 'Conv_134']
    end_node = ['Mul_23', 'Mul_45', 'Mul_67', 'Mul_89', 'Mul_111', 'Mul_133', 'Mul_155']
    transpose1_list = ['Transpose_2', 'Transpose_25', 'Transpose_47',
                       'Transpose_69', 'Transpose_91', 'Transpose_113', 'Transpose_135']
    transpose2_list = ['Transpose_15', 'Transpose_37', 'Transpose_59',
                       'Transpose_81', 'Transpose_103', 'Transpose_125', 'Transpose_147']
    fix_conv1d(args.input_path, args.output_path, beg_node, end_node, transpose1_list, transpose2_list)
