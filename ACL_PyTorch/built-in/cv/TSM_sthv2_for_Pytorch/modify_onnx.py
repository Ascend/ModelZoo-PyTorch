"""
Copyright 2023 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
"""
import argparse
import numpy as np
from auto_optimizer import OnnxGraph


def delete_sub(model, bs):
    sub_nodes = model.get_nodes('Sub')
    for node in sub_nodes:
        if node.inputs[0] == node.inputs[1]:
            next_slice_node = model.get_next_nodes(node.outputs[0])[0]
            pre_slice_node = model.get_prev_node(node.inputs[0])
            reshape_node = model.get_prev_node(pre_slice_node.inputs[0])
            # get the shape of the data
            shape = model[reshape_node.inputs[1]].value.copy()
            shape[0] = 6 * bs
            inputs = pre_slice_node.inputs[1:]
            inputs = [model[i].value for i in inputs]
            for i, axis in enumerate(inputs[2]):
                shape[axis] = (inputs[1][i] - inputs[0][i])//inputs[3][i]
            inputs = next_slice_node.inputs[1:]
            inputs = [model[i].value for i in inputs]
            for i, axis in enumerate(inputs[2]):
                shape[axis] = (inputs[1][i] - inputs[0][i])//inputs[3][i]
            # add a new initializer
            name = next_slice_node.outputs[0]
            model.add_initializer(name, np.zeros(shape, dtype=np.float32))
            # delete sub and slice nodes
            model.remove(node.name, {})
            model.remove(next_slice_node.name, {})
    return model


def merge_slice(model):
    slice_nodes = model.get_nodes('Slice')
    continuous_nodes = set()
    for node in slice_nodes:
        next_node = model.get_next_nodes(node.outputs[0])[0]
        if next_node.op_type == 'Slice':
            continuous_nodes.add((node, next_node))
    for i, nodes in enumerate(continuous_nodes):
        node_0 = nodes[0]
        node_1 = nodes[1]
        inputs_0 = node_0.inputs
        inputs_1 = node_1.inputs
        inputs = [inputs_0[0]]
        for j in range(1, len(inputs_0)):
            value_0 = model[inputs_0[j]].value
            value_1 = model[inputs_1[j]].value
            value = np.append(value_0, value_1)
            name = 'Initializer_' + str(i) + '_' + str(j)
            model.add_initializer(name, value)
            inputs.append(name)
        node_0.inputs = inputs
        model.remove(node_1.name)
    return model


def change_reshape(model):
    for node in model.get_nodes('Gemm'):
        next_node = model.get_next_nodes(node.outputs[0])[0]
        if next_node.op_type == 'Reshape':
            model[next_node.inputs[1]].value = np.array([-1, 48, 174])
    return model


def modify(input_model, output_model, batch_size):
    model = OnnxGraph.parse(input_model).simplify()
    model = delete_sub(model, batch_size)
    model = merge_slice(model)
    model = change_reshape(model)
    model.update_map()
    model.remove_unused_nodes()
    model.save(output_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--input_name', required=True, help='filepath of the original onnx model')
    parser.add_argument('-m2', '--output_name', required=True, help='filepath of the modified onnx model')
    parser.add_argument('-bs', '--batch_size', required=True, type=int, help='batch size of the model input')
    args = parser.parse_args()

    modify(args.input_name, args.output_name, args.batch_size)
