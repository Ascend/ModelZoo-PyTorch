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
import sys
import numpy as np
from onnx_tools.OXInterface.OXInterface import OXGraph


INPUT_NODE = 'image'
FIX_NODE = 'Relu_36'


def conv1d2conv2d(oxgraph, node_conv):
    """
    transfer conv1d parameters to conv2d
    :param oxgraph: input onnx graph
    :param node_conv: conv1d node to be transfered
    """
    if node_conv.get_op_type() != 'Conv':
        return
    node_conv.set_attribute(attr_name='dilations', attr_value=[1, 1])
    node_conv.set_attribute(attr_name='kernel_shape', attr_value=[1, 1])
    node_conv.set_attribute(attr_name='pads', attr_value=[0, 0, 0, 0])
    node_conv.set_attribute(attr_name='strides', attr_value=[1, 1])
    init_conv_w = oxgraph.get_oxinitializer_by_name(node_conv.input[1])
    init_conv_w.set_data(np.expand_dims(init_conv_w.get_data(), axis=2))


def adhoc_fix_multi_output(oxgraph, oxnode):
    """
    adhoc func for multi output for 'Relu35'
    insert sqeeze node before the second output
    return the first output
    :param oxgraph: input onnx graph
    :param oxnode: input onnx node(Relu35)
    """
    next_beg_nodes = oxgraph.get_next_oxnode(oxnode.get_name())
    first_out = next_beg_nodes[0]
    second_out = next_beg_nodes[1]
    if first_out.get_op_type() == 'Transpose':
        first_out, second_out = second_out, first_out
    squeeze_node_name = 'Squeeze_after_{}'.format(second_out.get_name())
    oxgraph.insert_node(bef_node_info_list=[oxnode.get_name()],
                        aft_node_info_list=[second_out.get_name()],
                        op_type='Squeeze',
                        op_name=squeeze_node_name)
    node_squeeze = oxgraph.get_oxnode_by_name(squeeze_node_name)
    node_squeeze.set_attribute(attr_name='axes', attr_value=[2])
    return [first_out]


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
        if next_beg_node.get_name() == FIX_NODE:
            next_beg_node = adhoc_fix_multi_output(oxgraph, next_beg_node)
        else:
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
    beg_nodes = ['Conv_3', 'Conv_35', 'Conv_72']
    end_nodes = ['Relu_8', 'Relu_45', 'Conv_74']
    fix_conv1d(input_path, save_path, beg_nodes, end_nodes)
