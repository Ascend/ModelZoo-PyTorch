import sys
import numpy as np
from onnx_tools.OXInterface.OXInterface import OXGraph


INPUT_NODE = 'mel'
FIX_NODE = '1'


def conv1d2conv2d(oxgraph, node_conv):
    """
    transfer conv1d parameters to conv2d
    :param oxgraph: input onnx graph
    :param node_conv: conv1d node to be transfered
    """
    if node_conv.get_op_type() != 'Conv':
        return
    node_conv.set_attribute(attr_name='dilations', attr_value=[1, 1])
    node_conv.set_attribute(attr_name='kernel_shape', attr_value=[1, 5])
    node_conv.set_attribute(attr_name='pads', attr_value=[0, 1, 0, 1])
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



def transfer_structure1(oxgraph, beg_node, end_node):

   
    next_beg_node = oxgraph.get_oxnode_by_name(oxnode_name=beg_node)
    while next_beg_node.get_name() != end_node:
        conv1d2conv2d(oxgraph, next_beg_node)
        print(next_beg_node.get_name())
        if next_beg_node.get_name() == FIX_NODE:
            next_beg_node = adhoc_fix_multi_output(oxgraph, next_beg_node)
        else:
            next_beg_node = oxgraph.get_next_oxnode(oxnode_name=next_beg_node.get_name())
        next_beg_node = next_beg_node[0]
    conv1d2conv2d(oxgraph, next_beg_node)

def change_conv2d(oxgraph):
    oxnode = oxgraph.get_oxnode_by_name('Conv_0')
    oxnode.set_attribute(attr_name='pads', attr_value=[0, 2, 0, 2])
    node_list = ['Conv_3', 'Conv_35', 'Conv_46', 'Conv_78', 'Conv_90', 'Conv_122']
    for node in node_list:
        oxnode = oxgraph.get_oxnode_by_name(node)
        oxnode.set_attribute(attr_name='pads', attr_value=[0, 0, 0, 0])

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

    beg_nodes = ['Conv_0', 'Conv_3', 'Conv_7', 'Conv_11', 'Conv_15', 'Conv_19', 'Conv_23', 'Conv_27', 'Conv_31', 'Conv_35', 'Conv_46', 'Conv_50', 'Conv_54', 'Conv_58', 'Conv_62', 'Conv_66', 'Conv_70', 'Conv_74', 'Conv_78', 'Conv_90', 'Conv_94', 'Conv_98', 'Conv_102', 'Conv_106', 'Conv_110', 'Conv_114', 'Conv_118', 'Conv_122', 'Conv_136']
    end_nodes = ['Relu_1', 'Relu_4', 'Relu_8', 'Relu_12', 'Relu_16', 'Relu_20', 'Relu_24', 'Relu_28', 'Relu_32', 'Relu_36', 'Relu_47', 'Relu_51', 'Relu_55', 'Relu_59', 'Relu_63', 'Relu_67', 'Relu_71', 'Relu_75', 'Relu_79', 'Relu_91', 'Relu_95', 'Relu_99', 'Relu_103', 'Relu_107', 'Relu_111', 'Relu_115', 'Relu_119', 'Relu_123', 'Relu_137']
    for idx, beg_node in enumerate(beg_nodes):
        end_node = end_nodes[idx]
        transfer_structure1(oxgraph, beg_node, end_node)
        
    change_conv2d(oxgraph)
    
    oxgraph.save_new_model(out_path)


if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    beg_nodes = ['Conv_0','Conv_3','Conv_46','Conv_90','Conv_136']
    end_nodes = ['Relu_1','Relu_36','Relu_79','Relu_123','Relu_137']
    fix_conv1d(input_path, save_path, beg_nodes, end_nodes)
    