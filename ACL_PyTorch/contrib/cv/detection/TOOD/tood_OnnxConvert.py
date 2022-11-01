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

# -*- coding: utf-8 -*-
import argparse
import os
import onnx


# change model
def change_model(args):
    """[change model and simplify model, If the height and width are lists,
     models of different input sizes are generated]

    Args:
        args ([argparse]): [change model and simplify model parameters]
    """
    model = onnx.load(args.input_name)
    model_nodes = model.graph.node
    name = 1 # node name
    for i in range(len(model_nodes)):
        if model_nodes[i].name in ["Add_1950", "Add_2000", "Add_2050", "Add_1900", "Add_1850", "Add_2200"]:
            old_node = model_nodes[i]
            newnode = onnx.helper.make_node(
            'Cast',
            name='Cast_new_{}'.format(name),
            inputs=[model_nodes[i].input[0]],
            to=onnx.TensorProto.INT32,
            outputs=[model_nodes[i].name + '_input']
            )
            model.graph.node.insert(i, newnode)
            old_node.input[0] = old_node.name + '_input'
            old_node.name = old_node.name + '_new'
            name += 1
    onnx.save(model, args.output_name)

if __name__ == "__main__":
    # change model parameters
    parser = argparse.ArgumentParser(description='change onnx model Expand and ScatterND \
                                     operator input float64 to float32. simplify model')
    parser.add_argument('--input_name', default='tood.onnx',
                        type=str, help='input onnx model name')
    parser.add_argument('--output_name', default='tood_convert.onnx',
                        type=str, help='output onnx model name')
    args = parser.parse_args()
    change_model(args)
    
