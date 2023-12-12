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


def add_flash_attention(model, fa_name):
    for i, node in enumerate(model.get_nodes('Softmax')):
        mul_node = model.get_prev_node(node.inputs[0])
        einsum_node = model.get_prev_node(mul_node.inputs[0])
        cast_node1 = model.get_prev_node(einsum_node.inputs[0])
        cast_node2 = model.get_prev_node(einsum_node.inputs[1])
        reshape_node1 = model.get_prev_node(cast_node1.inputs[0])
        reshape_node2 = model.get_prev_node(cast_node2.inputs[0])
        # move mul to q
        mul_node.inputs[0] = reshape_node1.outputs[0]
        # add flashattention
        new_node = model.add_node(f"FlashAttentionTik{i}", fa_name)
        inputs = [None, None, None]
        # input 0: q
        inputs[0] = mul_node.outputs[0]
        # input 1: k
        inputs[1] = reshape_node2.outputs[0]
        # input 2: v
        last_node = model.get_next_nodes(node.outputs[0])[0]
        inputs[2] = last_node.inputs[1]
        outputs = last_node.outputs
        new_node.inputs = inputs
        new_node.outputs = outputs
        model.remove(cast_node1.name, {})
        model.remove(cast_node2.name, {})
        model.remove(einsum_node.name, {})
        model.remove(node.name, {})
        model.remove(last_node.name, {})
        model.update_map()


def build_index(h, w, sy=2, sx=2):
    # random select one from a 2x2 block
    hsy = h // sy
    wsx = w // sx
    rand_idx = np.random.randint(sy * sx, size=(hsy, wsx))
        
    idx = np.ones((hsy, wsx, sy * sx), dtype=np.int64)
    for i in range(hsy):
        for j in range(wsx):
            idx[i, j][rand_idx[i, j]] = 0
    idx = idx.reshape(hsy, wsx, sy, sx).transpose(0, 2, 1, 3)
    idx_rand = idx.reshape(-1).argsort()
    index_a = np.sort(idx_rand[hsy * wsx:])
    index_b = np.sort(idx_rand[:hsy * wsx])
    return index_a, index_b


def get_block(model):
    # find self-attention block
    norms = []
    for node in model.get_nodes('Add'):
        next_nodes = model.get_next_nodes(node.outputs[0])
        if len(next_nodes) != 3:
            continue
        op_type = set(n.op_type for n in next_nodes)
        if len(op_type) == 1 and 'MatMul' in op_type:
            if model[node.inputs[1]].value.shape[0] == 320:
                norms.append(node)
    return norms


def find_nodes(model, node):
    prev_node = model.get_prev_node(node.inputs[0])
    while prev_node.op_type != 'Sub':
        prev_node = model.get_prev_node(prev_node.inputs[0])
    inp = prev_node.inputs[0]
    next_nodes = model.get_next_nodes(inp)
    for next_node in next_nodes:
        if next_node.op_type == 'Add':
            if next_node.inputs[0] == inp:
                out = next_node.inputs[1]
            else:
                out = next_node.inputs[0]
    return inp, out


def build_tome_block(model, i, inputs, inputs_un):
    # link merge to attn
    for node in model.get_next_nodes(inputs[1]):
        ind = 0
        for inp in node.inputs:
            if inp == inputs[1]:
                node.inputs[ind] = f'Concat_output{i}'
            ind += 1
    # norm block
    model.add_node(
        f'Mul{i}',
        'Mul',
        inputs=[inputs[0], inputs[0]],
        outputs=[f'Mul_output{i}']
    )
    model.add_node(
        f'ReduceSum{i}',
        'ReduceSum',
        inputs=[f'Mul_output{i}', 'reduce1/axes'],
        outputs=[f'ReduceSum_output{i}'],
        attrs={'keepdims': 1, 'noop_with_empty_axes': 1}
    )
    model.add_node(
        f'Sqrt{i}',
        'Sqrt',
        inputs=[f'ReduceSum_output{i}'],
        outputs=[f'Sqrt_output{i}']
    )
    model.add_node(
        f'Div{i}',
        'Div',
        inputs=[inputs[0], f'Sqrt_output{i}'],
        outputs=[f'Div_output{i}']
    )
    # compute similarity
    model.add_node(
        f'Gather_0_{i}',
        'Gather',
        inputs=[f'Div_output{i}', 'tome/Gather_index_a'],
        outputs=[f'Gather_0_output{i}'],
        attrs={'axis': 1}
    )
    model.add_node(
        f'Gather_1_{i}',
        'Gather',
        inputs=[f'Div_output{i}', 'tome/Gather_index_b'],
        outputs=[f'Gather_1_output{i}'],
        attrs={'axis': 1}
    )
    model.add_node(
        f'Transpose{i}',
        'Transpose',
        inputs=[f'Gather_1_output{i}'],
        outputs=[f'Transpose_output{i}'],
        attrs={'perm': [0, 2, 1]}
    )
    model.add_node(
        f'MatMul{i}',
        'MatMul',
        inputs=[f'Gather_0_output{i}', f'Transpose_output{i}'],
        outputs=[f'MatMul_output{i}']
    )
    model.add_node(
        f'FindMax{i}',
        'FindMax',
        inputs=[f'MatMul_output{i}'],
        outputs=[f'FindMax_output_0{i}', f'FindMax_output_1_{i}'],
        attrs={}
    )
    model.add_node(
        f'TopK{i}',
        'TopK',
        inputs=[f'FindMax_output_0{i}', 'tome/Topk_k'],
        outputs=[f'TopK_output_0{i}', f'TopK_output_1_{i}'],
        attrs={'axis': -1, 'largest': 1}
    )
    # split token
    model.add_node(
        f'Gather_2_{i}',
        'Gather',
        inputs=[inputs[1], 'tome/Gather_index_a'],
        outputs=[f'Gather_2_output{i}'],
        attrs={'axis': 1}
    )
    model.add_node(
        f'Gather_3_{i}',
        'Gather',
        inputs=[inputs[1], 'tome/Gather_index_b'],
        outputs=[f'Gather_3_output{i}'],
        attrs={'axis': 1}
    )
    model.add_node(
        f'Cast_0_{i}',
        'Cast',
        inputs=[f'Gather_2_output{i}'],
        outputs=[f'Cast_0_output{i}'],
        attrs={'to': 1}
    )
    model.add_node(
        f'Cast_1_{i}',
        'Cast',
        inputs=[f'Gather_3_output{i}'],
        outputs=[f'Cast_1_output{i}'],
        attrs={'to': 1}
    )
    # tome merge
    merge_inputs = [
        f'Cast_0_output{i}', 
        f'Cast_1_output{i}', 
        f'TopK_output_1_{i}', 
        f'FindMax_output_1_{i}'
    ]
    merge_outputs = [
        f'TomeMerged_output_0_{i}',
        f'TomeMerged_output_1_{i}',
        f'TomeMerged_output_2_{i}'
    ]
    model.add_node(
        f'TomeMerged{i}',
        'TomeMerged',
        inputs=merge_inputs,
        outputs=merge_outputs
    )
    model.add_node(
        f'ReduceSum_1_{i}',
        'ReduceSum',
        inputs=[f'TomeMerged_output_1_{i}', 'reduce2/axes'],
        outputs=[f'ReduceSum_1_output{i}'],
        attrs={'keepdims': 0, 'noop_with_empty_axes': 1}
    )
    model.add_node(
        f'ReduceSum_2_{i}',
        'ReduceSum',
        inputs=[f'TomeMerged_output_2_{i}', 'reduce2/axes'],
        outputs=[f'ReduceSum_2_output{i}'],
        attrs={'keepdims': 0, 'noop_with_empty_axes': 1}
    )
    model.add_node(
        f'Unsqueeze{i}',
        'Unsqueeze',
        inputs=[f'ReduceSum_2_output{i}', 'Unsqueeze/axes'],
        outputs=[f'Unsqueeze_output{i}']
    )
    model.add_node(
        f'Div_1_{i}',
        'Div',
        inputs=[f'ReduceSum_1_output{i}', f'Unsqueeze_output{i}'],
        outputs=[f'Div_1_output{i}']
    )
    model.add_node(
        f'Concat{i}',
        'Concat',
        inputs=[f'TomeMerged_output_0_{i}', f'Div_1_output{i}'],
        outputs=[f'Concat_output{i}'],
        attrs={'axis': 1}
    )
    # link unmerge to norm
    for node in model.get_next_nodes(inputs_un[0]):
        ind = 0
        for inp in node.inputs:
            if inp == inputs_un[0]:
                node.inputs[ind] = f'TomeUngerme_output{i}'
            ind += 1
    # add unmerge node
    unmerge_inputs = inputs_un + [f'TopK_output_1_{i}', f'FindMax_output_1_{i}']
    model.add_node(
        f'tome/TomeUnmerge{i}',
        'TomeUnmerge',
        inputs=unmerge_inputs,
        outputs=[f'TomeUngerme_output{i}']
    )
    model.update_map()


def insert_tome_block(model):
    bs = model['text'].shape[0]
    h, w = model['text'].shape[2:]
    index_a, index_b = build_index(h, w)
    # add initializer
    model.add_initializer('tome/Gather_index_a', index_a)
    model.add_initializer('tome/Gather_index_b', index_b)
    bs_index_a = np.tile(index_a.reshape(1, -1), [bs, 1])
    bs_index_b = np.tile(index_b.reshape(1, -1), [bs, 1])
    model.add_initializer('tome/index_a', bs_index_a)
    model.add_initializer('tome/index_b', bs_index_b)
    model.add_initializer('reduce1/axes', np.array([-1]))
    model.add_initializer('reduce2/axes', np.array([1]))
    model.add_initializer('Unsqueeze/axes', np.array([2]))
    model.add_initializer('tome/Topk_k', np.array([3456]))
    # find inputs
    norm_outs = get_block(model)
    for i, node in enumerate(norm_outs):
        norm_input, sa_output = find_nodes(model, node)
        inputs_0 = [norm_input] + node.outputs
        inputs_1 = [sa_output] + ['tome/index_a', 'tome/index_b']
        # add tome block
        build_tome_block(model, i, inputs_0, inputs_1)


def change_shapes_of_reshape(model):
    reshapes = []
    matmuls = model.get_nodes('MatMul')
    for node in matmuls:
        next_node1 = model.get_next_nodes(node.outputs[0])[0]
        if not next_node1.op_type == 'Reshape':
            continue
        next_node2 = model.get_next_nodes(next_node1.outputs[0])[0]
        if not next_node2.op_type == 'Transpose':
            continue
        next_node3 = model.get_next_nodes(next_node2.outputs[0])[0]
        if not next_node3.op_type == 'Reshape':
            continue
        reshapes.append(next_node1)
        reshapes.append(next_node3)

    for node in reshapes:
        shape_initializer = None
        shape_node = model.get_node(node.inputs[1], node_type=Initializer)
        if shape_node:
            shape_initializer = model[node.inputs[1]]
        else:
            shape_node = model.get_prev_node(node.inputs[1])
            if not shape_node.op_type == 'Identity':
                continue
            shape_initializer = model[shape_node.inputs[0]]

        shape = shape_initializer.value.copy()
        for i, size in enumerate(shape):
            if size == 4608:
                shape[i] = '-1'
        shape_initializer.value = shape


def replace_slice(model):
    slice_list = model.get_nodes('Slice')
    slice_pairs = [slice_list[i: i+2] for i in range(0, len(slice_list), 2)]
    for i, node in enumerate(slice_pairs):
        next_node = model.get_next_nodes(node[0].outputs[0])[0]
        if next_node.op_type == 'Mul':
            model.add_node(f'SliceTransGeluMul{i}', 'SliceTransGeluMul', inputs=[node[0].inputs[0]], outputs=next_node.outputs)
            model.remove(next_node.name, {})
        model.remove(node[0].name, {})
        model.remove(node[1].name, {})
    model.update_map()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/unet/unet.onnx",
        help="Path of the unet onnx model.",
    )
    parser.add_argument(
        "--new_model",
        type=str,
        default="models/unet/unet_md.onnx",
        help="Path to save the modified model",
    )
    return parser.parse_args()


def main():
    model = OnnxGraph.parse(args.model)
    add_flash_attention(model, 'FlashAttentionTik')
    insert_tome_block(model)
    change_shapes_of_reshape(model)
    replace_slice(model)
    model.remove_unused_nodes()
    model.save(args.new_model)


if __name__ == '__main__':
    args = parse_arguments()
    main()
