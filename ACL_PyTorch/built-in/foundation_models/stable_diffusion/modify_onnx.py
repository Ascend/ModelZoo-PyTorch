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


def del_add(model):
    init = [n.name for n in model.get_nodes('Initializer')]
    for node in model.get_nodes('Add'):
        if 'attn' in node.name and node.inputs[1] in init:
            value = model[node.inputs[1]].value
            if (value == 0).all():
                model.remove(node.name)

            
def add_flash_attention(model, fa_name, soc_type):
    for node in model.get_nodes('Mul'):
        name = node.name
        if soc_type == 1:
            flag = 'attn' in name
        else:
            flag = 'attn1' in name
        if flag:
            matmul = model[name[:-3] + 'to_q/MatMul']
            reshape = model[name[:-3] + 'Reshape']
            if soc_type == 2 and model[reshape.inputs[1]].value[1] != 4096:
                continue
            softmax_node = model.get_next_nodes(node.outputs[0])[0]
            if soc_type == 1:
                # move mul to q
                softmax_node.inputs[0] = node.inputs[0]
                node.inputs[0] = matmul.outputs[0]
                reshape.inputs[0] = node.outputs[0]

            # add flashattention
            new_node = model.add_node(name[:-3] + fa_name, fa_name)
            inputs = [None, None, None]
            # input 0: q
            if soc_type == 1:
                matmul_node = model.get_prev_node(softmax_node.inputs[0])
            if soc_type == 2:
                matmul_node = model.get_prev_node(node.inputs[0])
            inputs[0] = matmul_node.inputs[0]
            # input 1: k
            transpose_node = model.get_prev_node(matmul_node.inputs[1])
            inputs[1] = transpose_node.inputs[0]
            # input 2: v
            cast_node = model.get_next_nodes(softmax_node.outputs[0])[0]
            last_node = model.get_next_nodes(cast_node.outputs[0])[0]
            inputs[2] = last_node.inputs[1]
            # output
            outputs = last_node.outputs
            # update link
            new_node.inputs = inputs
            new_node.outputs = outputs
            
            model.remove(matmul_node.name, {})
            model.remove(transpose_node.name, {})
            model.remove(softmax_node.name, {})
            model.remove(cast_node.name, {})
            model.remove(last_node.name, {})
    model.update_map()
    for node in model.get_nodes(fa_name):
        for _ in range(soc_type):
            for i in range(3):
                prev_node = model.get_prev_node(node.inputs[i])
                model.remove(prev_node.name)
            next_node = model.get_next_nodes(node.outputs[0])[0]
            model.remove(next_node.name)
        if soc_type == 2:
            name = node.name.replace(fa_name, 'Cast')
            cast = model.add_node(name, 'Cast', attrs={'to': 1})
            model.insert_node(node.name, cast)


def change_input_type(model):
    model.remove('t')
    model.add_input('t', 'int32', [1])
    model.inputs[1], model.inputs[2] = model.inputs[2], model.inputs[1]


def replace_slice(model):
    # find pairs of slice
    slice_pair = []
    for node in model.get_nodes('Slice'):
        if node.name[:-2] == '_1':
            slice_pair.append((node.name[:-2], node.name))
    # replace
    for pair in slice_pair:
        name = pair[0][:-5] + 'Split'
        data = pair[0].inputs[0]
        if pair[0].inputs[1] == pair[1].inputs[2]:
            outputs = pair[1].outputs + pair[0].outputs
        if pair[1].inputs[1] == pair[0].inputs[2]:
            outputs = pair[0].outputs + pair[1].outputs
        axes = pair[0].inputs[3]
        axis = model[axes].value[0]
        model.add_node(name, 'Split', inputs=[data], outputs=outputs, attrs={'axis':axis})
        model.remove(pair[0].name, {})
        model.remove(pair[1].name, {})
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


def build_tome_block(model, name, inputs, inputs_un):
    # link merge to attn
    for node in model.get_next_nodes(inputs[1]):
        ind = 0
        for inp in node.inputs:
            if inp == inputs[1]:
                node.inputs[ind] = name + 'Concat_output'
            ind += 1
    # norm block
    model.add_node(
        name + 'Mul',
        'Mul',
        inputs=[inputs[0], inputs[0]],
        outputs=[name + 'Mul_output']
    )
    model.add_node(
        name + 'ReduceSum',
        'ReduceSum',
        inputs=[name + 'Mul_output'],
        outputs=[name + 'ReduceSum_output'],
        attrs={'axes': [-1], 'keepdims': 1}
    )
    model.add_node(
        name + 'Sqrt',
        'Sqrt',
        inputs=[name + 'ReduceSum_output'],
        outputs=[name + 'Sqrt_output']
    )
    model.add_node(
        name + 'Div',
        'Div',
        inputs=[inputs[0], name + 'Sqrt_output'],
        outputs=[name + 'Div_output']
    )
    # compute similarity
    model.add_node(
        name + 'Gather_0',
        'Gather',
        inputs=[name + 'Div_output', 'tome/Gather_index_a'],
        outputs=[name + 'Gather_0_output'],
        attrs={'axis': 1}
    )
    model.add_node(
        name + 'Gather_1',
        'Gather',
        inputs=[name + 'Div_output', 'tome/Gather_index_b'],
        outputs=[name + 'Gather_1_output'],
        attrs={'axis': 1}
    )
    model.add_node(
        name + 'Transpose',
        'Transpose',
        inputs=[name + 'Gather_1_output'],
        outputs=[name + 'Transpose_output'],
        attrs={'perm': [0, 2, 1]}
    )
    model.add_node(
        name + 'MatMul',
        'MatMul',
        inputs=[name + 'Gather_0_output', name + 'Transpose_output'],
        outputs=[name + 'MatMul_output']
    )
    model.add_node(
        name + 'FindMax',
        'FindMax',
        inputs=[name + 'MatMul_output'],
        outputs=[name + 'FindMax_output_0', name + 'FindMax_output_1'],
        attrs={}
    )
    model.add_node(
        name + 'TopK',
        'TopK',
        inputs=[name + 'FindMax_output_0', 'tome/Topk_k'],
        outputs=[name + 'TopK_output_0', name + 'TopK_output_1'],
        attrs={'axis': -1, 'largest': 1}
    )
    # split token
    model.add_node(
        name + 'Gather_2',
        'Gather',
        inputs=[inputs[1], 'tome/Gather_index_a'],
        outputs=[name + 'Gather_2_output'],
        attrs={'axis': 1}
    )
    model.add_node(
        name + 'Gather_3',
        'Gather',
        inputs=[inputs[1], 'tome/Gather_index_b'],
        outputs=[name + 'Gather_3_output'],
        attrs={'axis': 1}
    )
    model.add_node(
        name + 'Cast_0',
        'Cast',
        inputs=[name + 'Gather_2_output'],
        outputs=[name + 'Cast_0_output'],
        attrs={'to': 1}
    )
    model.add_node(
        name + 'Cast_1',
        'Cast',
        inputs=[name + 'Gather_3_output'],
        outputs=[name + 'Cast_1_output'],
        attrs={'to': 1}
    )
    # tome merge
    merge_inputs = [
        name + 'Cast_0_output', 
        name + 'Cast_1_output', 
        name + 'TopK_output_1', 
        name + 'FindMax_output_1'
    ]
    merge_outputs = [
        name + 'TomeMerged_output_0',
        name + 'TomeMerged_output_1',
        name + 'TomeMerged_output_2'
    ]
    model.add_node(
        name + 'TomeMerged',
        'TomeMerged',
        inputs=merge_inputs,
        outputs=merge_outputs
    )
    model.add_node(
        name + 'ReduceSum_1',
        'ReduceSum',
        inputs=[name + 'TomeMerged_output_1'],
        outputs=[name + 'ReduceSum_1_output'],
        attrs={'axes': [1], 'keepdims': 0}
    )
    model.add_node(
        name + 'ReduceSum_2',
        'ReduceSum',
        inputs=[name + 'TomeMerged_output_2'],
        outputs=[name + 'ReduceSum_2_output'],
        attrs={'axes': [1], 'keepdims': 0}
    )
    model.add_node(
        name + 'Unsqueeze',
        'Unsqueeze',
        inputs=[name + 'ReduceSum_2_output'],
        outputs=[name + 'Unsqueeze_output'],
        attrs={'axes': [2]}
    )
    model.add_node(
        name + 'Div_1',
        'Div',
        inputs=[name + 'ReduceSum_1_output', name + 'Unsqueeze_output'],
        outputs=[name + 'Div_1_output']
    )
    model.add_node(
        name + 'Concat',
        'Concat',
        inputs=[name + 'TomeMerged_output_0', name + 'Div_1_output'],
        outputs=[name + 'Concat_output'],
        attrs={'axis': 1}
    )
    # link unmerge to norm
    for node in model.get_next_nodes(inputs_un[0]):
        ind = 0
        for inp in node.inputs:
            if inp == inputs_un[0]:
                node.inputs[ind] = name + 'TomeUngerme_output'
            ind += 1
    # add unmerge node
    unmerge_inputs = inputs_un + [name + 'TopK_output_1', name + 'FindMax_output_1']
    model.add_node(
        name + 'tome/TomeUnmerge',
        'TomeUnmerge',
        inputs=unmerge_inputs,
        outputs=[name + 'TomeUngerme_output']
    )
    model.update_map()


def insert_tome_block(model, max_num):
    bs = model['latent_model_input'].shape[0]
    h, w = model['latent_model_input'].shape[2:]
    index_a, index_b = build_index(h, w)
    # add initializer
    model.add_initializer('tome/Gather_index_a', index_a)
    model.add_initializer('tome/Gather_index_b', index_b)
    bs_index_a = np.tile(index_a.reshape(1, -1), bs)
    bs_index_b = np.tile(index_b.reshape(1, -1), bs)
    model.add_initializer('tome/index_a', bs_index_a)
    model.add_initializer('tome/index_b', bs_index_b)
    model.add_initializer('tome/Topk_k', np.array([3072]))
    # get reshape nodes
    reshapes = model.get_nodes('Reshape')
    # find inputs
    norm_outs = get_block(model)[:max_num]
    for node in norm_outs:
        name = node.name.rsplit('/', 2)[0] + '/attn1/'
        norm_input, sa_output = find_nodes(model, node)
        inputs_0 = [norm_input] + node.outputs
        inputs_1 = [sa_output] + ['tome/index_a', 'tome/index_b']
        # add tome block
        build_tome_block(model, name.replace('attn', 'tome'), inputs_0, inputs_1)
        # change shape of reshape
        for reshape in reshapes:
            if name in reshape.name:
                shape = model[reshape.inputs[1]].value.copy()
                ind = 0
                for size in shape:
                    if size == 4096:
                        shape[ind] = '-1'
                    ind += 1
                model[reshape.inputs[1]].value = shape


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
    parser.add_argument(
        "--FA_soc", 
        choices=["None", "Duo", "A2"],
        default="None", 
        help="Type of FA operator.",
    )
    parser.add_argument(
        "--TOME_num",
        type=int,
        default=0,
        help="Number of TOME used in the model",
    )
    return parser.parse_args()


def main():
    model = OnnxGraph.parse(args.model)
    del_add(model)
    if args.FA_soc == 'Duo':
        add_flash_attention(model, 'FlashAttentionTik', soc_type=1)
    elif args.FA_soc == 'A2':
        add_flash_attention(model, 'UnpadFlashAttentionMix', soc_type=2)
    if args.TOME_num:
        insert_tome_block(model, args.TOME_num)
    change_input_type(model)
    replace_slice(model)
    model.remove_unused_nodes()
    model.save(args.new_model)


if __name__ == '__main__':
    args = parse_arguments()
    main()