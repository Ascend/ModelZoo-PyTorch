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
from auto_optimizer import OnnxInitializer as Initializer
from auto_optimizer.graph_refactor.interface import Node


def add_BatchMultiClassNMS(onnx_graph):
    """Replace nonmaxsuppression with BatchMultiClassNMS"""

    # initailize old_nodes list
    old_nodes = []
    nms_node = onnx_graph.get_nodes('NonMaxSuppression')[0]
    pre_concat_node = onnx_graph.get_prev_node(nms_node.inputs[0])
    pre_slice_node = onnx_graph.get_prev_node(nms_node.inputs[1])
    for output_name in [*pre_concat_node.outputs, *pre_slice_node.outputs]:
        next_nodes = onnx_graph.get_next_nodes(output_name)
        if next_nodes:
            old_nodes.extend(next_nodes)

    # delete downstream nodes
    while old_nodes:
        to_del_node = old_nodes.pop(0)
        if to_del_node.name == 'Add_labels':
            continue
        if not onnx_graph.get_node(to_del_node.name, Node):
            continue
        for output_name in to_del_node.outputs:
            next_nodes = onnx_graph.get_next_nodes(output_name)
            if next_nodes:
                old_nodes.extend(next_nodes)

        onnx_graph.remove(to_del_node.name)

    # Add unsqueeze cast operators to satisfy batchmulticlassnms 
    # input requirements
    unsqueeze_node = onnx_graph.add_node(
        'Unsqueeze_new_0', 
        'Unsqueeze',
        inputs=[''],
        outputs=['Unsqueeze_new_0_out_0'],
        attrs={'axes': [2]}
    )
  
    cast_node = onnx_graph.add_node(
        'Cast_new_0', 
        'Cast',
        inputs=[''],
        outputs=['Add_new_0_out_0'],
        attrs={'to': 7}
    )

    bnms_node = onnx_graph.add_node(
        'BatchMultiClassNMS_new_0', 
        'BatchMultiClassNMS',
        inputs=['', ''], outputs=['', '', ''],
        attrs=dict(
            iou_threshold=0.5,
            max_size_per_class=200,
            max_total_size=200,
            score_threshold=0.05
        )
    )
        
    onnx_graph.insert_node(pre_concat_node.name, unsqueeze_node, refer_index=0, 
                           mode='after')
    onnx_graph.insert_node('Add_labels', cast_node, refer_index=0,  
                           mode='before')
    onnx_graph.connect_node(bnms_node, ['Unsqueeze_new_0', pre_slice_node.name],
                            ['bboxes', 'scores', 'Cast_new_0'])
    onnx_graph.remove_unused_nodes()
    return onnx_graph


def process_pattern(pattern, onnx_graph, prev_node1):
    """Find the corresponding position of the operator in the graph by pattern"""
    pattern_nodes = []
    for op_type in pattern[1:]:
        prev_node1 = onnx_graph.get_prev_node(prev_node1.inputs[0])
        if prev_node1.op_type != op_type:
            flag = False
            break
        pattern_nodes.append(prev_node1)
    return pattern_nodes


def replace_sub_add_mul(onnx_graph):
    """Substituting with fewer operators achieves the same function"""
    concat_nodes = onnx_graph.get_nodes('Concat')
    old_nodes = []
    # Use the concat operator to find other operators to delete
    for concat_node in concat_nodes:
        prev_nodes = [onnx_graph.get_prev_node(input_name) 
                      for input_name in concat_node.inputs]
        if len(prev_nodes) != 4 or not all(prev_node.op_type == 'Unsqueeze' 
                                           for prev_node in prev_nodes):
            continue
        old_nodes.extend(prev_nodes)

        flag = True
        for unsqueeze_node in prev_nodes:
            prev_node = onnx_graph.get_prev_node(unsqueeze_node.inputs[0])
            if prev_node.op_type not in ['Sub', 'Add']:
                flag = False
                break
            old_nodes.append(prev_node)
            add_oldnode, mul_oldnode = None, None
            
            # Determine the specific position in the diagram that matches this pattern
            for input_name in prev_node.inputs:
                pattern1 = ['Squeeze', 'Slice', 'Slice', 'Slice', 'Add']
                pattern2 = ['Mul', 'Squeeze', 'Slice', 'Slice', 'Slice', 'Mul']
                prev_node1 = onnx_graph.get_prev_node(input_name)
                pattern = None
                if prev_node1.op_type == pattern1[0]:
                    pattern = pattern1
                elif prev_node1.op_type == pattern2[0]:
                    pattern = pattern2
                else:
                    flag = False
                    break
                old_nodes.append(prev_node1)
                pattern_nodes = process_pattern(pattern, onnx_graph, prev_node1)
                if pattern_nodes is None:
                    flag = False
                    break
                old_nodes.extend(pattern_nodes)
                last_node = old_nodes.pop(-1)
                if last_node.op_type == 'Add':
                    add_oldnode = last_node
                if last_node.op_type == 'Mul':
                    mul_oldnode = last_node

        # Add equivalent substitution operators
        if flag:
            for node in old_nodes:
                onnx_graph.remove(node.name)
            new_mul = onnx_graph.add_node('new_mul', 'Mul',inputs=['', ''], 
                                          outputs=['new_mul_out'])
            new_add = onnx_graph.add_node('new_add', 'Add',inputs=['', ''], 
                                          outputs=['new_add_out'])
            new_sub = onnx_graph.add_node('new_sub', 'Sub',inputs=['', ''], 
                                          outputs=['new_sub_out'])
            Mul_ini = onnx_graph.add_initializer('Mul_ini', np.array([0.5],  
                                                 dtype=np.float32))
            onnx_graph.connect_node(
                new_mul,
                [f'{mul_oldnode.name}', 'Mul_ini'],
                ['new_add:1', 'new_sub:1']
            )
            onnx_graph.connect_node(
                new_add,
                [f'{add_oldnode.name}', 'new_mul'],
                [f'{concat_node.name}:1']
            )
            onnx_graph.connect_node(
                new_sub,
                [f'{add_oldnode.name}','new_mul'],
                [f'{concat_node.name}:0']
            )

            concat_node.inputs = concat_node.inputs[:2]
            onnx_graph.remove_unused_nodes()
    return onnx_graph


def replace_slice_with_split(onnx_graph):
    """Replace the two sequences of Slice nodes with a Split node."""
    
    transpose_nodes = onnx_graph.get_nodes('Transpose')
    for transpose_node in transpose_nodes:
        next_nodes = onnx_graph.get_next_nodes(transpose_node.outputs[0])
        if len(next_nodes) != 2 or not \
            all(next_node.op_type == 'Slice' for next_node in next_nodes):
            continue

        left_node = onnx_graph.get_next_nodes(next_nodes[0].outputs[0])[0]
        right_node = onnx_graph.get_next_nodes(next_nodes[1].outputs[0])[0]
        if not (left_node.op_type == right_node.op_type == 'Slice'):
            continue
        left_node = onnx_graph.get_next_nodes(left_node.outputs[0])[0]
        right_node = onnx_graph.get_next_nodes(right_node.outputs[0])[0]
        if not (left_node.op_type == right_node.op_type == 'Slice'):
            continue
        left_node = onnx_graph.get_next_nodes(left_node.outputs[0])[0]
        right_node = onnx_graph.get_next_nodes(right_node.outputs[0])[0]
        new_split = onnx_graph.add_node('new_split', 'Split', 
                                    attrs={'split': 2, 'axis':2}, 
                                    outputs=['split_out1', 'split_out2'])
        onnx_graph.connect_node(new_split, [f'{transpose_node.name}'], 
                            [f'{left_node.name}', 
                            f'{right_node.name}'])
    onnx_graph.remove_unused_nodes()
    return onnx_graph


def delete_redundant_transpose(onnx_graph):
    """
    ... -> TransPose -> Slice -> Slice -> Slice -> BatchMultiClassNMS -> ...

    In the example above, do two modifications:
        1. After generating the OM model, the Transpose node is 
           unused, so need to delete it.
        2. Correspondingly, exchange start index of last two Slice nodes.
    """
    
    bnms_newnode = onnx_graph.get_nodes('BatchMultiClassNMS')

    slice0_node = onnx_graph.get_prev_node(bnms_newnode[0].inputs[1])
    slice1_node = onnx_graph.get_prev_node(slice0_node.inputs[0])

    constant0 = onnx_graph.get_node(slice0_node.inputs[1], Initializer)
    constant1 = onnx_graph.get_node(slice1_node.inputs[1], Initializer)

    slice2_node = onnx_graph.get_prev_node(slice1_node.inputs[0])
    transpose_oldnode = onnx_graph.get_prev_node(slice2_node.inputs[0])

    constant1.value = np.array([0])
    constant0.value = np.array([1])

    onnx_graph.remove(transpose_oldnode.name)
    return onnx_graph

def modify_onnx(input_onnx, output_onnx):
    g = OnnxGraph.parse(input_onnx)
    g = add_BatchMultiClassNMS(g)
    g = replace_sub_add_mul(g)
    g = replace_slice_with_split(g)
    g = delete_redundant_transpose(g)
    g.save(output_onnx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='modify onnx model.')
    parser.add_argument('input_onnx', type=str, help='path to input onnx file.')
    parser.add_argument('output_onnx', type=str, 
                        help='path to save modified onnx model.')
    args = parser.parse_args()
    modify_onnx(args.input_onnx, args.output_onnx)