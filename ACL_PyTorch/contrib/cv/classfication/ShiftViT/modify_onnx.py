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


import sys
import onnx
from onnx import TensorProto


def modify_onnx(input_onnx, output_onnx):
    
    model = onnx.load(input_onnx)
    
    # create a new node
    Y = onnx.helper.make_tensor('Y', onnx.TensorProto.FLOAT, [1], [1])
    model.graph.initializer.append(Y)
    pow_node = onnx.helper.make_node(
        'Pow',
        inputs=['1824', 'Y'],
        outputs=['pow_insert_out'],
        name='Pow_insert'
    )
    
    # find the insert location of new node
    insert_idx = -1
    for i, node in enumerate(model.graph.node):
        if node.name == 'Mul_1463' and node.input[0] == '1824':
            insert_idx = i
            node.input[0] = 'pow_insert_out'
            break
    if insert_idx == -1:
        raise Exception("can not find the insert location.")
        
    model.graph.node.insert(insert_idx, pow_node)
    onnx.save(model, output_onnx)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser('modify the onnx model.')
    parser.add_argument('-i', '--input-onnx-path', type=str,
                        help='path to original onnx model.')
    parser.add_argument('-o', '--output-onnx-path', type=str,
                        help='path to save modified onnx model.')
    args = parser.parse_args()

    modify_onnx(args.input_onnx_path, args.output_onnx_path)


