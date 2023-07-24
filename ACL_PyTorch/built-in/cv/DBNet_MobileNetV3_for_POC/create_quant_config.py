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
import os
import stat

from auto_optimizer import OnnxGraph


def need_skip_conv(onnx_graph, conv_node):
    # check shape
    output_channel = onnx_graph.get_value_info(conv_node.outputs[0]).shape[1]
    if output_channel % 16 != 0:
        return False

    # check structure
    next_nodes = onnx_graph.get_next_nodes(conv_node.outputs[0])
    if len(next_nodes) != 1 or next_nodes[0].op_type != 'BatchNormalization':
        return False
    next_nodes = onnx_graph.get_next_nodes(next_nodes[0].outputs[0])
    if len(next_nodes) != 2:
        return False
    next_nodes_dict = {node.op_type: node for node in next_nodes}
    if set(next_nodes_dict.keys()) != set(['Add', 'Mul']):
        return False
    next_nodes = onnx_graph.get_next_nodes(next_nodes_dict['Add'].outputs[0])
    if len(next_nodes) != 1 or next_nodes[0].op_type != 'Clip':
        return False

    return True


def need_skip_convtranspose(onnx_graph, convt_node):
    # check shape
    output_channel = onnx_graph.get_value_info(convt_node.outputs[0]).shape[1]
    if output_channel % 16 != 0:
        return False

    # check structure
    next_nodes = onnx_graph.get_next_nodes(convt_node.outputs[0])
    if len(next_nodes) != 1 or next_nodes[0].op_type != 'BatchNormalization':
        return False

    return True


def find_skip_nodes(onnx_path, save_path):
    g = OnnxGraph.parse(onnx_path)
    num_nodes = len(g.nodes)

    skip_node_names = []
    for i, node in enumerate(g.nodes):

        # Under certain conditions, Dequant will destroy Conv/ConvTranspose 
        # fusion, and further lead to performance lower. So, need to skip these 
        # nodes when quantization.
        if node.op_type == 'Conv' and need_skip_conv(g, node):
            skip_node_names.append(node.name)
            continue
        if node.op_type == 'ConvTranspose' and need_skip_convtranspose(g, node):
            skip_node_names.append(node.name)
            continue

        # skip last layer to reduce precision loss
        if num_nodes - i < 10 and node.op_type in ['Conv', 'ConvTranspose']:
            skip_node_names.append(node.name)
            continue

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(save_path, flags, modes), 'w') as fout:
        for node_name in skip_node_names:
            fout.write(f'skip_layers : "{node_name}"\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description='create quantization config for AMCT.')
    parser.add_argument('input_onnx', type=str, help='path to onnx file.')
    parser.add_argument('output_config', type=str, 
                        help='path to save quantization config.')
    args = parser.parse_args()
    find_skip_nodes(args.input_onnx, args.output_config)
