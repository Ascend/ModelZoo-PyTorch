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

import argparse
import numpy as np
from auto_optimizer import OnnxGraph


def add_pad(model, bs):
    # add initialzer
    concat_pad = 'pad_concat'
    gather_pad = 'pad_gather'
    slice_starts = 'start'
    slice_ends = 'end'
    slice_axes = 'axis'
    slice_steps = 'step'
    pad_concat = np.zeros((bs, 11, 384), dtype=np.float32)
    model.add_initializer(concat_pad, pad_concat)
    pad_gatehr = np.zeros((bs, 6, 11, 64), dtype=np.float32)
    model.add_initializer(gather_pad, pad_gatehr)
    model.add_initializer(slice_starts, np.array([0]))
    model.add_initializer(slice_ends, np.array([197]))
    model.add_initializer(slice_axes, np.array([2]))
    model.add_initializer(slice_steps, np.array([1]))

    for node in model.get_nodes('Concat'):
        next_node = model.get_next_nodes(node.outputs[0])[0]
        if next_node.op_type in ['Add', 'ReduceMean']:
            node.inputs.append(concat_pad)

    for node in model.get_nodes('Slice'):
        if model[node.inputs[2]].value > 197:
            node.inputs[2] = 'end'

    i = 0
    inputs = [slice_starts, slice_ends, slice_axes, slice_steps]
    for node in model.get_nodes('Transpose'):
        next_nodes = model.get_next_nodes(node.outputs[0])
        if len(next_nodes) == 3 and next_nodes[0].op_type == 'Gather':
            i += 1
            for j, next_node in enumerate(next_nodes):
                slice_name = 'Slice_{}_{}'.format(i, j)
                new_slice = model.add_node(slice_name, 'Slice')
                model.insert_node(next_node.name, new_slice)
                new_slice.inputs.extend(inputs)
        
    for node in model.get_nodes('Softmax'):
        next_node = model.get_next_nodes(node.outputs[0])[0]
        i += 1
        concat_name = 'Concat_new_' + str(i)
        new_concat = model.add_node(concat_name, 'Concat', attrs={'axis': 2})
        model.insert_node(next_node.name, new_concat)
        new_concat.inputs.append(gather_pad)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True,
                        help='filename of original onnx model')
    parser.add_argument('--save_path', required=True,
                        help='filename of modified onnx model')
    parser.add_argument('--batch_size', required=True, type=int,
                        help='batch size of onnx model')
    args = parser.parse_args()

    model_1 = OnnxGraph.parse(args.model_path)
    model_2 = add_pad(model_1, args.batch_size)
    model_2.save(args.save_path)
