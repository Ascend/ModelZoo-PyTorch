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

from auto_optimizer import OnnxGraph, OnnxNode


def convert_conv1d_to_conv2d(graph: OnnxGraph, conv: OnnxNode) -> bool:
    attrs = ('dilations', 'kernel_shape', 'strides')
    for attr in attrs:
        if attr in conv.attrs.keys():
            val = conv[attr][0]
            conv[attr] = [1, val]

    if 'pads' in conv.attrs.keys():
        pds = conv['pads'][0]
        conv['pads'] = [0, pds, 0, pds]

    conv_w = graph[conv.inputs[1]].value
    conv_w = np.expand_dims(conv_w, axis=-2)
    graph[conv.inputs[1]].value = conv_w
    return True


def validate_insert_mode(insert_mode: str) -> bool:
    if isinstance(insert_mode, str) and insert_mode in ('before', 'after'):
        return True
    else:
        raise ValueError(f'Invalid insert_mode: "{insert_mode}", which should be one of ["before", "after"].')


def insert_unsqueeze(graph: OnnxGraph, node: OnnxNode, attrs: dict, insert_mode: str) -> bool:
    if not attrs.get('axes'):
        raise RuntimeError('Insert unsqueeze failed, missing the attribute "axes".')

    validate_insert_mode(insert_mode)
    op_name = f'Unsqueeze_{insert_mode}_{node.name}'
    unsqueeze = graph.add_node(op_name, 'Unsqueeze', attrs = attrs)
    graph.insert_node(node.name, unsqueeze, mode=insert_mode)



def insert_squeeze(graph: OnnxGraph, node: OnnxNode, attrs: dict, insert_mode: str) -> bool:
    if not attrs.get('axes'):
        raise RuntimeError('Insert squeeze failed, missing the attribute "axes".')

    validate_insert_mode(insert_mode)
    op_name = f'Squeeze_{insert_mode}_{node.name}'
    squeeze = graph.add_node(op_name, 'Squeeze', attrs = attrs)
    graph.insert_node(node.name, squeeze, mode=insert_mode)


def optimize(model_path: str, save_path: str) -> None:
    graph = OnnxGraph.parse(model_path)

    for conv in graph.get_nodes("Conv")[:-1]:
        convert_conv1d_to_conv2d(graph, conv)

    first_conv = graph.get_nodes("Conv")[0]
    insert_unsqueeze(graph, first_conv, {'axes': [2]}, 'before')

    first_transpose = graph.get_nodes("Transpose")[0]
    insert_squeeze(graph, first_transpose, {'axes': [2]}, 'before')

    first_reshape = graph.get_nodes("Reshape")[0]
    graph[first_reshape.inputs[1]].value = np.array([0, 512, 1, -1])

    first_mul = graph.get_nodes("Mul")[0]
    graph[first_mul.inputs[1]].value = np.reshape(graph[first_mul.inputs[1]].value, (1, 512, 1, 1))

    first_add = graph.get_nodes("Add")[0]
    graph[first_add.inputs[1]].value = np.reshape(graph[first_add.inputs[1]].value, (1, 512, 1, 1))

    graph.save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', help='the input path of ONNX model to be modified',
                         default="wav2vec2.onnx")
    parser.add_argument('--output_model_path', help='the path of ONNX model to be saved',
                         default="wav2vec2_modified.onnx")
    args = parser.parse_args()

    optimize(args.input_model_path, save_path=args.output_model_path)

    print("Done.")
