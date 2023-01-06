# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#
#!/usr/bin/env python3.8

import argparse
from auto_optimizer import OnnxGraph


def delete_concat(graph):
    '''Delete redundant concat. e.g.:
      2      data        2    data
      |      Mul    =>    |   Mul
      |  /  \  |           \  /
    Concat  Concat        Concat
    '''
    concat_nodes = graph.get_nodes('Concat')
    node_input = dict()
    for node in concat_nodes:
        inp = tuple(node.inputs)
        if inp in node_input:
            remain_node =  node_input[inp]
            next_node = graph.get_next_nodes(node.outputs[0])[0]
            next_node.inputs[0] = remain_node.outputs[0]
            graph.remove(node.name, {})
        else:
            node_input[inp] = node
    graph.update_map()
    return graph


def replace_matmul(graph):
    '''replace matmul with gemm. e.g.:
    concat-matmul-slice -> concat-squeeze-gemm-unsqueeze-split
    reshape-matmul-add -> reashape-squeeze-gemm-unsqueeze-add
    '''
    def set_gemm(graph):
        gemm_name = 'Gemm_' + node.name.split('_')[1]
        graph.add_node(gemm_name, 'Gemm', inputs=node.inputs, outputs=node.outputs)
        graph.remove(node.name, {})

        sq_name = 'Squeeze_before_' + gemm_name
        sq_node = graph.add_node(sq_name, 'Squeeze', attrs={'axes': [1]})
        graph.insert_node(gemm_name, sq_node, 0, 'before')

        unsq_name = 'Unsqueeze_after_' + gemm_name
        unsq_node = graph.add_node(unsq_name, 'Unsqueeze', attrs={'axes': [1]})
        graph.insert_node(gemm_name, unsq_node)
       
    matmul_nodes = graph.get_nodes('MatMul')
    for node in matmul_nodes:
        prev_node0 = graph.get_prev_node(node.inputs[0])
        prev_node1 = graph.get_prev_node(node.inputs[1])
        prev_node = prev_node0 if prev_node0 else prev_node1
        next_node = graph.get_next_nodes(node.outputs[0])[0]
        if prev_node.op_type == 'Concat' and next_node.op_type == 'Slice':
            set_gemm(graph)
        elif prev_node.op_type == 'Reshape' and next_node.op_type == 'Add':
            set_gemm(graph)
    return graph
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_onnx', required=True, help='filename of the original onnx model')
    parser.add_argument('-o', '--output_onnx', required=True, help='filename of the modified onnx model')
    args = parser.parse_args()

    g = OnnxGraph.parse(args.input_onnx)
    g = delete_concat(g)
    g = replace_matmul(g)
    g.remove_unused_nodes()
    g.infershape()
    g.save(args.output_onnx)

    print('Successfully saved modified model.')