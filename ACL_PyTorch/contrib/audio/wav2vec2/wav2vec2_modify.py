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


import sys
import argparse
import numpy as np
from magiconnx import OnnxGraph

def conv1d2conv2d(graph, node_conv):
    if node_conv == 'Conv_1':
        graph[node_conv]['dilations'] = [1, 1]
        m = graph[node_conv]['kernel_shape'][0]
        graph[node_conv]['kernel_shape'] = [1, m]
        graph[node_conv]['pads'] = [0, 0, 0, 0]
        graph[node_conv]['strides'] = [1, 5]
        wight = graph[graph[node_conv].node.input[1]].value
        graph[graph[node_conv].node.input[1]].value = np.expand_dims(wight, axis=2)
        graph[graph['Reshape_3'].node.input[1]].value = np.array([0, 512, 1, -1])
        Mul = graph[graph['Mul_9'].node.input[1]].value
        graph[graph['Mul_9'].node.input[1]].value = np.squeeze(Mul)
        Add = graph[graph['Add_10'].node.input[1]].value
        graph[graph['Add_10'].node.input[1]].value = Add.reshape(1, 512, 1, 1)
    else:
        graph[node_conv]['dilations'] = [1, 1]
        m = graph[node_conv]['kernel_shape'][0]
        graph[node_conv]['kernel_shape'] = [1, m]
        graph[node_conv]['pads'] = [0, 0, 0, 0]
        graph[node_conv]['strides'] = [1, 2]
        wight = graph[graph[node_conv].node.input[1]].value
        graph[graph[node_conv].node.input[1]].value = np.expand_dims(wight, axis=2)

def transfer_structure(graph, beg_node, end_node):
    if beg_node != 'Conv_1':
        squeeze_name = end_node + 'squeeze'
        OnnxGraph.add_node(graph, squeeze_name, 'Squeeze', attrs={'axes': [2]})
        graph.insert_node(end_node, graph[squeeze_name], mode='after')
    if beg_node != 'Conv_19':
        unsqueeze_name = beg_node + 'unsqueeze'
        OnnxGraph.add_node(graph, unsqueeze_name, 'Unsqueeze', attrs={'axes': [2]})
        graph.insert_node(beg_node, graph[unsqueeze_name], mode='before')
    conv1d2conv2d(graph, beg_node)

def fix_conv1d(model_path, out_path, beg_list, end_list):
    graph = OnnxGraph(model_path)
    for idx, beg_node in enumerate(beg_list):
        end_node = end_list[idx]
        transfer_structure(graph, beg_node, end_node)
    graph.save(out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', help='the input path of ONNX model to be modified',
                         default="wav2vec2.onnx")
    parser.add_argument('--output_model_path', help='the path of ONNX model to be saved',
                         default="wav2vec2_modified.onnx")

    args = parser.parse_args()
    beg_node = ['Conv_19', 'Conv_28', 'Conv_37', 'Conv_46', 'Conv_55', 'Conv_64', 'Conv_1']
    end_node = ['Mul_27', 'Mul_36', 'Mul_45', 'Mul_54', 'Mul_63', 'Mul_72', 'Mul_18']

    fix_conv1d(args.input_model_path, args.output_model_path, beg_node, end_node)
