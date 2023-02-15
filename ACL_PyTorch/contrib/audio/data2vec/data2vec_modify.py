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


import argparse
import numpy as np
from auto_optimizer import OnnxGraph


def conv1d2conv2d(model):
    conv_nodes = model.get_nodes('Conv')
    for node in conv_nodes:
        attrs = node.attrs
        dil = attrs['dilations'][0]
        ks = attrs['kernel_shape'][0]
        pads = attrs['pads']
        stride = attrs['strides'][0]
        attrs['dilations'] = [1, dil]
        attrs['kernel_shape'] = [1, ks]
        attrs['pads'] = [0, pads[0], 0, pads[1]]
        attrs['strides'] = [1, stride]
        name = node.inputs[1]
        weights = model[name].value
        weights = np.expand_dims(weights, axis=-2)
        model[name].value = weights
    return model


def change_perm(model):
    trans_nodes = model.get_nodes('Transpose')
    for node in trans_nodes:
        next_node = model.get_next_nodes(node.outputs[0])[0]
        if next_node.op_type == 'ReduceMean':
            node.attrs['perm'] = [0, 2, 3, 1]
        elif next_node.op_type == 'Div':
            node.attrs['perm'] = [0, 3, 1, 2]
        elif next_node.op_type == 'Add':
            node.attrs['perm'] = [0, 2, 3, 1]
        elif next_node.op_type == 'Conv':
            node.attrs['perm'] = [0, 3, 1, 2]
    return model


def change_dim(model):
    first_node = model.get_next_nodes('modelInput')[0]
    if first_node.op_type == 'Unsqueeze':
        first_node.attrs['axes'] = [1, 2]
    trans_nodes = model.get_nodes('Transpose')
    for node in trans_nodes:
        next_node = model.get_next_nodes(node.outputs[0])[0]
        if next_node.op_type == 'Add':
            sq_node = model.add_node('Squeeze_conv', 'Squeeze', attrs={'axes':[1]})
            model.insert_node(next_node.name, sq_node)
    return model


def change_conv(input_model, output_model):
    model = OnnxGraph.parse(input_model)
    model = conv1d2conv2d(model)
    model = change_perm(model)
    model = change_dim(model)
    model.infershape()
    model.save(output_model)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--input_name', required=True, help='filepath of the original onnx model')
    parser.add_argument('-m2', '--output_name', required=True, help='filepath of the modified onnx model')
    args = parser.parse_args()

    input_name = args.input_name
    output_name = args.output_name
    change_conv(input_name, output_name)