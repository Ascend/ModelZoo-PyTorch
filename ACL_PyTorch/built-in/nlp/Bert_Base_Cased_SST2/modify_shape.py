# Copyright 2023 Huawei Technologies Co., Ltd
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
import argparse
import numpy as np
from auto_optimizer import OnnxGraph


def dynamic_add(model_ori, model):
    node_1 = model.get_next_nodes('input_ids')[0]
    node_1 = model.get_next_nodes(node_1.outputs[0])[0]
    node_2 = model.get_next_nodes(node_1.outputs[0])[0]
    input_1 = node_1.inputs[1]
    input_2 = node_2.inputs[1]
    model.remove(input_1, {})
    model.remove(input_2, {})
    model.add_initializer('num1', np.ones(1).astype(np.int64))
    model.add_initializer('num0', np.zeros(1).astype(np.int64))
    model.add_node(
        'Shape_new',
        'Shape',
        inputs=['input_ids'],
        outputs=['input_shape']
    )
    model.add_node(
        'Gather_new', 
        'Gather',
        attrs={'axis': 0}, 
        inputs=['input_shape', 'num1'],
        outputs=['seq_len']
    )
    model.add_node(
        'Unsqueeze_new',
        'Unsqueeze', 
        attrs={'axes': [0]},
        inputs=['seq_len'],
        outputs=['len']
    )

    # input for add_39
    model.add_initializer('sequence0', np.zeros((1, 512)).astype(np.int64))
    token_embeddings = 'bert.embeddings.token_type_embeddings.weight'
    model.add_initializer(
        token_embeddings,
        model_ori[token_embeddings].value
    )
    model.add_node(
        'Slice_new_1',
        'Slice',
        inputs=['sequence0', 'num0', 'len', 'num1', 'num1'],
        outputs=['sentence']
    )
    model.add_node(
        'Gather_new_1',
        'Gather',
        inputs=[token_embeddings, 'sentence'],
        outputs=[input_1]
    )

    # input for add_41
    position_ids = 'bert.embeddings.position_ids'
    model.add_initializer(
        position_ids, 
        model_ori[position_ids].value
    )
    position_embeddings = 'bert.embeddings.position_embeddings.weight'
    model.add_initializer(
        position_embeddings,
        model_ori[position_embeddings].value
    )
    model.add_node(
        'Slice_new_2',
        'Slice',
        inputs=[position_ids, 'num0', 'len', 'num1', 'num1'],
        outputs=['position']
    )
    model.add_node(
        'Gather_new_2',
        'Gather',
        inputs=[position_embeddings, 'position'],
        outputs=[input_2]
    )

    return model


def fix_reshape(model, bs):
    for node in model.get_nodes('Reshape'):
        shape = node.inputs[1]
        value = model[shape].value.copy()
        if value.shape[0] == 4:
            value[1] = -1
        if value.shape[0] == 2:
            value[0] = -1
        if value.shape[0] == 3:
            value[0] = bs
            value[1] = -1
        model[shape].value = value
    return model


def add_expand(model, bs):
    model.add_initializer('num-1', np.array([-1]).astype(np.int64))
    model.add_initializer('batch', np.array([bs]).astype(np.int64))
    model.add_node(
        'Concat_new',
        'Concat',
        attrs={'axis': 0},
        inputs=['batch', 'num1', 'seq_len', 'seq_len'],
        outputs=['expand_shape']
    )
    model.add_node(
        'Reshape_new',
        'Reshape',
        inputs=['expand_shape', 'num-1'],
        outputs=['new_expand_shape']
    )
    for node in model.get_nodes('Expand'):
        node.inputs[1] = 'new_expand_shape'
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--original_onnx',
        required=True,
        help='path of the original onnx model before simplifier'
    )
    parser.add_argument(
        '--modified_onnx',
        required=True,
        help='path of the modified onnx model after modify_onnx'
    )
    parser.add_argument(
        '--save_path',
        required=True,
        help='path to save the new onnx model'
    )
    parser.add_argument(
        '--batch_size',
        required=True,
        type=int,
        help='the batch size of the model inputs'
    )
    args = parser.parse_args()

    model_1 = OnnxGraph.parse(args.original_onnx)
    model_2 = OnnxGraph.parse(args.modified_onnx)
    new_model = dynamic_add(model_1, model_2)
    new_model = fix_reshape(new_model, args.batch_size)
    new_model = add_expand(new_model, args.batch_size)
    new_model.update_map()
    new_model.save(args.save_path)