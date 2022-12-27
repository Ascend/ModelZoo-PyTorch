# Copyright 2022 Huawei Technologies Co., Ltd
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

from magiconnx import OnnxGraph


def modify_max_pool(graph: OnnxGraph) -> OnnxGraph:
    """modify MaxPool3D operator, insert Unsqueeze and Squeeze operator arround it."""
    unsqueeze1 = graph.add_node('unsqueeze1', 'Unsqueeze', {'axes': [0]},
                                inputs=['65'], outputs=['65_unsq'])
    max_pool_2 = graph['MaxPool_2']
    max_pool_2.inputs = ['65_unsq']
    max_pool_2.outputs = ['66_mp']
    squeeze1 = graph.add_node('squeeze1', 'Squeeze', {'axes': [0]},
                              inputs=['66_mp'], outputs=['66'])

    unsqueeze2 = graph.add_node('unsqueeze2', 'Unsqueeze', {'axes': [0]},
                                inputs=['75'], outputs=['75_unsq'])
    max_pool_2 = graph['MaxPool_11']
    max_pool_2.inputs = ['75_unsq']
    max_pool_2.outputs = ['76_mp']
    squeeze1 = graph.add_node('squeeze2', 'Squeeze', {'axes': [0]},
                              inputs=['76_mp'], outputs=['76'])

    unsqueeze3 = graph.add_node('unsqueeze3', 'Unsqueeze', {'axes': [0]},
                                inputs=['94'], outputs=['94_unsq'])
    max_pool_2 = graph['MaxPool_28']
    max_pool_2.inputs = ['94_unsq']
    max_pool_2.outputs = ['95_mp']
    squeeze1 = graph.add_node('squeeze3', 'Squeeze', {'axes': [0]},
                              inputs=['95_mp'], outputs=['95'])

    return graph


def modify_reduce_mean(graph: OnnxGraph) -> OnnxGraph:
    """modify ReduceMean operator, add axes attribute"""
    reduce_mean_38_new = graph.add_node('reduce_mean_38_new', 'ReduceMean',
                                        {'axes': [0, 1, 2, 3], 'keepdims': 0},
                                        inputs=['106'], outputs=['107'])
    graph.del_node('ReduceMean_38')
    graph['Div_39'].inputs = ['104', '107']

    reduce_mean_45_new = graph.add_node('reduce_mean_45_new', 'ReduceMean',
                                        {'axes': [0, 1, 2, 3], 'keepdims': 0},
                                        inputs=['113'], outputs=['114'])
    graph.del_node('ReduceMean_45')
    graph['Div_46'].inputs = ['111', '114']

    reduce_mean_52_new = graph.add_node('reduce_mean_52_new', 'ReduceMean',
                                        {'axes': [0, 1, 2, 3], 'keepdims': 0},
                                        inputs=['120'], outputs=['121'])
    graph.del_node('ReduceMean_52')
    graph['Div_53'].inputs = ['118', '121']

    reduce_mean_56_new = graph.add_node('reduce_mean_56_new', 'ReduceMean',
                                        {'axes': [0, 1, 2, 3], 'keepdims': 0},
                                        inputs=['124'], outputs=['125'])
    graph.del_node('ReduceMean_56')
    graph['Div_57'].inputs = ['101', '125']

    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', dest='onnx', default='./LPRNet_sim.onnx',
                        help='onnx model path')
    parser.add_argument('--output', dest='output', default='./LPRNet_mod.onnx',
                        help='modified onnx model store path')
    args = parser.parse_args()

    graph = OnnxGraph(args.onnx)
    graph = modify_max_pool(graph)
    graph = modify_reduce_mean(graph)
    graph.save(args.output)
