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
from auto_optimizer import OnnxGraph


def move_add_to_convtranspose(onnx_graph):
    """
    ConvTranspose(x, W)
            |              convert to     
            v             ------------>  ConvTranspose(x, W, B=add_val)
      Add(x, add_val)
    """

    convt_nodes = onnx_graph.get_nodes('ConvTranspose')
    for convt_node in convt_nodes:
        # if ConvTranspose node has bias input, continue.
        if len(convt_node.inputs) >= 3:
            continue
        next_nodes = onnx_graph.get_next_nodes(convt_node.outputs[0])
        # if ConvTranspose node has multiple outputs, or conv_transpose node's 
        # next node is not Add node, continue.
        if len(next_nodes) != 1 or next_nodes[0].op_type != 'Add':
            continue

        # convert Add node's value to ConvTranspose node's bias
        add_node = next_nodes[0]
        add_value_index = 1 - add_node.inputs.index(convt_node.outputs[0])
        add_value_name = add_node.inputs[add_value_index]
        convt_bias_name = f'{convt_node.name}_bias'
        convt_bias = onnx_graph.add_initializer(
            convt_bias_name, onnx_graph[add_value_name].value.flatten())
        convt_node.inputs.append(convt_bias_name)

        # delete unused node
        onnx_graph.remove(add_node.name)
        onnx_graph.remove(add_value_name)

    return onnx_graph


def modify_onnx(input_onnx, output_onnx):
    g = OnnxGraph.parse(input_onnx)
    g.inputs[0].shape = ["-1", 3, 736, 1280]
    g.infershape()
    g = move_add_to_convtranspose(g)
    g.update_map()
    g.save(output_onnx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description='modify onnx model.')
    parser.add_argument('input_onnx', type=str, help='path to input onnx file.')
    parser.add_argument('output_onnx', type=str, 
                        help='path to save modified onnx model.')
    args = parser.parse_args()
    modify_onnx(args.input_onnx, args.output_onnx)
