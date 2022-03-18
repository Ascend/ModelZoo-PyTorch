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
    h_list, w_list, bs_list = args.height, args.width, args.batch_size
    for k in range(len(h_list)):
        h, w, bs = h_list[k], w_list[k], bs_list[k]
        input_model = args.input_name + '_{}x{}.onnx'.format(h, w)
        output_model = args.output_name + '_{}x{}.onnx'.format(h, w)
        os.system('rm -rf {}'.format(output_model))
        model = onnx.load(input_model)
        onnx.checker.check_model(model)
        model_nodes = model.graph.node
        name = 1 # node name
        for i in range(len(model_nodes)):
            if model_nodes[i].name in ['Expand_70', 'Expand_105', 'Expand_140', 'Expand_175']:
                old_node = model_nodes[i]
                newnode = onnx.helper.make_node(
                'Cast',
                name='Cast_new_{}'.format(name),
                inputs=[model_nodes[i].input[0]],
                to=1,
                outputs=[model_nodes[i].name + '_input']
                )
                model.graph.node.insert(i, newnode)
                old_node.input[0] = old_node.name + '_input'
                old_node.name = old_node.name + '_new'
                name += 1
            if model_nodes[i].name in ['Expand_77', 'Expand_85', 'Expand_112', 'Expand_120', 
                                        'Expand_147', 'Expand_155', 'Expand_182', 'Expand_190']:
                old_node = model_nodes[i]
                newnode = onnx.helper.make_node(
                'Cast',
                name='Cast_new_{}'.format(name),
                inputs=[model_nodes[i].input[0]],
                to=6,
                outputs=[model_nodes[i].name + '_input']
                )
                model.graph.node.insert(i, newnode)
                old_node.input[0] = old_node.name + '_input'
                old_node.name = old_node.name + '_new'
                name += 1 
            if model_nodes[i].name in ['ScatterND_95', 'ScatterND_130', 'ScatterND_165', 'ScatterND_200']:
                old_node = model_nodes[i]
                newnode = onnx.helper.make_node(
                'Cast',
                name='Cast_new_{}'.format(name),
                inputs=[model_nodes[i].input[1]],
                to=7,
                outputs=[model_nodes[i].name + '_input']
                )
                model.graph.node.insert(i, newnode)
                old_node.input[1] = old_node.name + '_input'
                old_node.name = old_node.name + '_new'
                name += 1

        onnx.checker.check_model(model)
        onnx.save(model, output_model)
        simplified_model_name = args.simplified_name + '_{}x{}.onnx'.format(h, w)
        os.system('rm -rf {}'.format(simplified_model_name))
        os.system('python3.7 -m onnxsim --input-shape="{},3,{},{}" {} {}'\
                    .format(bs, h, w, output_model, simplified_model_name))
    

if __name__ == "__main__":
    # change model parameters
    parser = argparse.ArgumentParser(description='change onnx model Expand and ScatterND \
                                     operator input float64 to float32. simplify model')
    parser.add_argument('--input_name', default='rcf_bs1',
                        type=str, help='input onnx model name')
    parser.add_argument('--output_name', default='rcf_bs1_change',
                        type=str, help='output onnx model name')
    parser.add_argument('--simplified_name', default='rcf_bs1_change_sim',
                        type=str, help='simplified onnx model name')
    parser.add_argument('--batch_size', nargs='+',
                        type=int, help='batch size')
    parser.add_argument('--height', nargs='+',
                        type=int, help='input height')
    parser.add_argument('--width', nargs='+',
                        type=int, help='input width')
    args = parser.parse_args()
    change_model(args)
    