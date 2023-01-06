# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import numpy as np
from auto_optimizer import OnnxGraph


def modify_onnx(model_path, save_path):
    g = OnnxGraph.parse(model_path)
    
    if not find_pattern(g):
        print('Can not find pattern: Mul -> Add -> GAP -> Flatten -> Gemm')
    else:
        print('The model is successfully modified.')
    
    g.save(save_path)


def find_pattern(graph):
    gemms = graph.get_nodes('Gemm')
    for gemm in gemms:
        flatten = graph.get_prev_node(gemm.inputs[0])
        if flatten.op_type != 'Flatten':
            return False
        gap = graph.get_prev_node(flatten.inputs[0])
        if gap.op_type != 'GlobalAveragePool':
            return False
        add = graph.get_prev_node(gap.inputs[0])
        if add.op_type != 'Add':
            return False
        mul = graph.get_prev_node(add.inputs[0])
        if mul.op_type != 'Mul':
            return False
        
        # insert Pow before Mul to interrupt fusion pass
        pow_new = graph.add_node('Pow_new', 'Pow')
        pow_ini = graph.add_initializer('Pow_ini', np.array([1]).astype('float32'))
        graph.insert_node(mul.name, pow_new, 0, 'before')
        pow_new.inputs.append('Pow_ini')
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        'modify model with auto_optimizer')
    parser.add_argument('-i', '--input-onnx', type=str, 
                        required=True, help='path to onnx file before modification')
    parser.add_argument('-o', '--output-onnx', type=str, 
                        required=True, help='path to onnx file after modification')
    args = parser.parse_args()

    modify_onnx(args.input_onnx, args.output_onnx)