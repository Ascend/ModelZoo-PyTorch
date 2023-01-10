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
import numpy as np

from auto_optimizer import OnnxGraph


def modify_max_pool_1(graph: OnnxGraph) -> OnnxGraph:
    """ simplify maxpool3d-1 """
    graph.add_node(name='MaxPool_2_new', op_type='MaxPool', 
                   inputs=['65'], outputs=['66'], 
                   attrs={'ceil_mode': np.int64(0), 
                          'kernel_shape': np.int64([3, 3]),
                          'pads': np.int64([0, 0, 0, 0]), 
                          'strides': np.int64([1, 1])})
    graph.remove(name='MaxPool_2', mapping={})

    return graph


def modify_max_pool_2(graph: OnnxGraph) -> OnnxGraph:
    """ optimize maxpool3d-2 """
    conv_9_w = graph[graph['Conv_9'].inputs[1]].value
    conv_9_w_even = conv_9_w[::2, ::, ::, ::]

    conv_9_b = graph[graph['Conv_9'].inputs[2]].value
    conv_9_b_even = conv_9_b[::2]

    graph.add_initializer(name='Conv_9_w_even', value=conv_9_w_even)
    graph.add_initializer(name='Conv_9_b_even', value=conv_9_b_even)

    graph.add_node(name='Conv_9_even', op_type='Conv', 
                   inputs=['72', 'Conv_9_w_even', 'Conv_9_b_even'], outputs=['conv_9_even'], 
                   attrs={'dilations': np.int64([1, 1]), 
                          'kernel_shape': np.int64([1, 1]), 
                          'pads': np.int64([0, 0, 0, 0]),
                          'strides': np.int64([1, 1])})

    graph.add_node(name='Relu_10_even', op_type='Relu', 
                   inputs=['conv_9_even'], outputs=['relu_10_even'])
    
    graph.add_node(name='MaxPool_11_new', op_type='MaxPool', 
                   inputs=['relu_10_even'], outputs=['76'], 
                   attrs={'ceil_mode': np.int64(0), 
                          'kernel_shape': np.int64([3, 3]),
                          'pads': np.int64([0, 0, 0, 0]), 
                          'strides': np.int64([1, 2])})
    graph.remove(name='MaxPool_11', mapping={})

    return graph


def modify_max_pool_3(graph: OnnxGraph) -> OnnxGraph:
    """ optimize maxpool3d-3 """
    conv_26_w = graph[graph['Conv_26'].inputs[1]].value
    conv_26_w_even = conv_26_w[::4, ::, ::, ::]

    conv_26_b = graph[graph['Conv_26'].inputs[2]].value
    conv_26_b_even = conv_26_b[::4]

    graph.add_initializer(name='Conv_26_w_even', value=conv_26_w_even)
    graph.add_initializer(name='Conv_26_b_even', value=conv_26_b_even)

    graph.add_node(name='Conv_26_even', op_type='Conv', 
                   inputs=['91', 'Conv_26_w_even', 'Conv_26_b_even'], outputs=['conv_26_even'], 
                   attrs={'dilations': np.int64([1, 1]), 
                          'kernel_shape': np.int64([1, 1]), 
                          'pads': np.int64([0, 0, 0, 0]), 
                          'strides': np.int64([1, 1])})

    graph.add_node(name='Relu_27_even', op_type='Relu', 
                   inputs=['conv_26_even'], outputs=['relu_27_even'])
    
    graph.add_node(name='MaxPool_28_new', op_type='MaxPool', 
                   inputs=['relu_27_even'], outputs=['95'], 
                   attrs={'ceil_mode': np.int64(0),
                          'kernel_shape': np.int64([3, 3]),
                          'pads': np.int64([0, 0, 0, 0]),  
                          'strides': np.int64([1, 2])})
    graph.remove(name='MaxPool_28',  mapping={})

    return graph


def modify_max_pool(graph: OnnxGraph) -> OnnxGraph:
    """ optimize all maxpool """
    graph = modify_max_pool_1(graph)
    graph = modify_max_pool_2(graph)
    graph = modify_max_pool_3(graph)

    return graph


def modify_reduce_mean(graph: OnnxGraph) -> OnnxGraph:
    """modify ReduceMean operator, add axes attribute"""
    graph.add_node(name='reduce_mean_38_new', op_type='ReduceMean', 
                   inputs=['106'], outputs=['107'],
                   attrs={'axes': np.int64([0, 1, 2, 3]), 'keepdims': np.int64(0)})
    graph.remove(name='ReduceMean_38', mapping={})

    graph.add_node(name='reduce_mean_45_new', op_type='ReduceMean', 
                   inputs=['113'], outputs=['114'],
                   attrs={'axes': np.int64([0, 1, 2, 3]), 'keepdims': np.int64(0)})
    graph.remove(name='ReduceMean_45', mapping={})

    graph.add_node(name='reduce_mean_52_new', op_type='ReduceMean', 
                   inputs=['120'], outputs=['121'],
                   attrs={'axes': np.int64([0, 1, 2, 3]), 'keepdims': np.int64(0)})
    graph.remove(name='ReduceMean_52', mapping={})

    graph.add_node(name='reduce_mean_56_new', op_type='ReduceMean', 
                   inputs=['124'], outputs=['125'],
                   attrs={'axes': np.int64([0, 1, 2, 3]), 'keepdims': np.int64(0)})
    graph.remove(name='ReduceMean_56', mapping={})

    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', dest='onnx', default='./LPRNet.onnx',
                        help='onnx model path')
    parser.add_argument('--output', dest='output', default='./LPRNet_mod.onnx',
                        help='modified onnx model store path')
    args = parser.parse_args()

    graph = OnnxGraph.parse(args.onnx)
    graph = modify_max_pool(graph)
    graph = modify_reduce_mean(graph)
    graph.save(args.output)
