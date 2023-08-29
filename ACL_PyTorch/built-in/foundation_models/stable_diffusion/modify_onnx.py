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

import sys
from auto_optimizer import OnnxGraph


def del_add():
    init = [n.name for n in model.get_nodes('Initializer')]
    for node in model.get_nodes('Add'):
        if 'attn' in node.name and node.inputs[1] in init:
            value = model[node.inputs[1]].value
            if (value == 0).all():
                model.remove(node.name)

            
def add_flash_attention(fa_name, soc_type):
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


def change_input_type():
    model.remove('t')
    model.add_input('t', 'int32', [1])
    model.inputs[1], model.inputs[2] = model.inputs[2], model.inputs[1]


def replace_slice():
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
        

if __name__ == '__main__':
    model = OnnxGraph.parse(sys.argv[1])
    soc = sys.argv[3]
    del_add()
    if soc == 'Duo':
        add_flash_attention(fa_name='FlashAttentionTik', soc_type=1)
    elif soc == 'A2':
        add_flash_attention(fa_name='UnpadFlashAttentionMix', soc_type=2)
    change_input_type()
    replace_slice()
    model.remove_unused_nodes()
    model.save(sys.argv[2])