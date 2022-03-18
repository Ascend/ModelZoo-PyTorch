# Copyright 2021 Huawei Technologies Co., Ltd
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
import numpy as np
from onnx_tools.OXInterface.OXInterface import OXGraph


INPUT_NODE = 'video'

def conv1d2conv2d(oxgraph, node_conv):
    """
    transfer conv1d parameters to conv2d
    :param oxgraph: input onnx graph
    :param node_conv: conv1d node to be transfered
    """
    if node_conv.get_op_type() != 'Conv':
        return
    if node_conv.get_name() == 'Conv_0':
        node_conv.set_attribute(attr_name='dilations', attr_value=[1,1])
        node_conv.set_attribute(attr_name='kernel_shape', attr_value=[1,3])
        node_conv.set_attribute(attr_name='pads', attr_value=[0, 1, 0, 1])
        node_conv.set_attribute(attr_name='strides', attr_value=[1,1])
    if node_conv.get_name() == 'Conv_2':
        node_conv.set_attribute(attr_name='dilations', attr_value=[1,1])
        node_conv.set_attribute(attr_name='kernel_shape', attr_value=[1,3])
        node_conv.set_attribute(attr_name='pads', attr_value=[0, 1, 0, 1])
        node_conv.set_attribute(attr_name='strides', attr_value=[1,1])
    if node_conv.get_name() == 'Conv_4':
        node_conv.set_attribute(attr_name='dilations', attr_value=[1,1])
        node_conv.set_attribute(attr_name='kernel_shape', attr_value=[1,1])
        node_conv.set_attribute(attr_name='pads', attr_value=[0, 0, 0, 0])
        node_conv.set_attribute(attr_name='strides', attr_value=[1,1])
    
    init_conv_w = oxgraph.get_oxinitializer_by_name(node_conv.input[1])
    init_conv_w.set_data(np.expand_dims(init_conv_w.get_data(), axis=2))

def transfer_structure(oxgraph, beg_node, end_node):
    """
    transfer process:
    1. insert unsqueeze node before beg node
    2. insert squeeze node after end node
    3. transfer conv1d paramters for conv2d
    :param oxgraph: input onnx graph
    :param beg_node: beg node name for searched structure
    :param end_node: end node name for searched structure
    """
    previous_beg_node = oxgraph.get_previous_oxnode(oxnode_name=beg_node)
    if not previous_beg_node:
        previous_beg_node = INPUT_NODE
    else:
        previous_beg_node = previous_beg_node[0].get_name()
    next_end_node = oxgraph.get_next_oxnode(oxnode_name=end_node)
    unsqueeze_node_name = 'Unsqueeze_before_{}'.format(beg_node)
    squeeze_node_name = 'Squeeze_after_{}'.format(end_node)
    next_end_node = next_end_node[0].get_name()

    oxgraph.insert_node(bef_node_info_list=[previous_beg_node],
                        aft_node_info_list=[beg_node],
                        op_type='Unsqueeze',
                        op_name=unsqueeze_node_name)
    oxgraph.insert_node(bef_node_info_list=[end_node],
                        aft_node_info_list=[next_end_node],
                        op_type='Squeeze',
                        op_name=squeeze_node_name)
    node_unsqueeze = oxgraph.get_oxnode_by_name(unsqueeze_node_name)
    node_unsqueeze.set_attribute(attr_name='axes', attr_value=[2])
    node_squeeze = oxgraph.get_oxnode_by_name(squeeze_node_name)
    node_squeeze.set_attribute(attr_name='axes', attr_value=[2])

    next_beg_node = oxgraph.get_oxnode_by_name(oxnode_name=beg_node)
    while next_beg_node.get_name() != end_node:
        conv1d2conv2d(oxgraph, next_beg_node)
        next_beg_node = oxgraph.get_next_oxnode(oxnode_name=next_beg_node.get_name())
        next_beg_node = next_beg_node[0]
    conv1d2conv2d(oxgraph, next_beg_node)


def fix_conv1d(model_path, out_path, beg_list, end_list):
    """
    main process for fixing conv1d
    :param model_path: input onnx model path
    :param out_path: out fixed onnx model path
    :param beg_list: beg node names for searched structure
    :param end_list: end node names for searched structure
    """
    oxgraph = OXGraph(model_path)
    for idx, beg_node in enumerate(beg_list):
        end_node = end_list[idx]
        transfer_structure(oxgraph, beg_node, end_node)
    oxgraph.save_new_model(out_path)
    

if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    beg_nodes = ['Conv_0']
    end_nodes = ['Conv_4']
    fix_conv1d(input_path, save_path, beg_nodes, end_nodes)
    