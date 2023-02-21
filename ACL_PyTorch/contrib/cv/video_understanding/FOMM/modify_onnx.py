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
import numpy as np
import argparse
from auto_optimizer import OnnxGraph


def cast32_to_64(model):
    for node in model.get_nodes('Cast'):
        if node.attrs['to'] == 7:
            node.attrs['to'] = 6
            next_nodes = model.get_next_nodes(node.outputs[0])
            add_nodes = [n for n in next_nodes if n.op_type == 'Add']
            if add_nodes[0].inputs == add_nodes[1].inputs:
                repeat_out = add_nodes[1].outputs[0]
                unique_out = add_nodes[0].outputs[0]
                for n in model.get_next_nodes(repeat_out):
                    inputs = n.inputs
                    for i in range(len(inputs)):
                        if inputs[i] == repeat_out:
                            inputs[i] = unique_out
                model.remove(add_nodes[1].name, {})
    for init in model.get_nodes('Initializer'):
        if init.value.dtype == np.int64:
            nodes = model.get_next_nodes(init.name)
            types = {node.op_type for node in nodes}
            change_types = {'Add', 'Less', 'Greater', 'Mul'}
            if len(change_types & types):
                init.value = np.int32(init.value)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', default='taichi-gen-bs1.onnx', help='filepath of the original onnx model')
    parser.add_argument('--output_name', default='taichi-gen-bs1_new.onnx', help='filepath of the modified onnx model')
    args = parser.parse_args()
    input_name = args.input_name
    output_name = args.output_name

    model = OnnxGraph.parse(input_name)
    model = model.simplify()
    model = cast32_to_64(model)
    model.save(output_name)