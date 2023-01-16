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
from auto-optimizer import OnnxGraph

def remove_zero_weights(model, bs):
    tanh_nodes = model.get_nodes('Tanh')
    remove_nodes = []
    for node in tanh_nodes:
        next_node = model.get_next_nodes(node.outputs[0])[0]
        if next_node.op_type == 'Slice':
            nodes = [None, None, None]
            while next_node:
                next_node = model.get_next_nodes(next_node.outputs[0])[-1]
                if next_node.op_type == 'Squeeze':
                    final_node = model.get_next_nodes(next_node.outputs[0])[0]
                    nodes[1] = final_node.name
                    break
            prev_node = model.get_prev_node(node.inputs[0])
            while prev_node:
                if prev_node.op_type == 'Add':
                    inp = prev_node.inputs[1]
                else:
                    inp = prev_node.inputs[0]
                prev_node = model.get_prev_node(inp)
                if prev_node.op_type == 'Unsqueeze':
                    nodes[0] = prev_node.name
                    prev_node = model.get_prev_node(prev_node.inputs[0])
                    nodes[2] = prev_node.name
                    break
            remove_nodes.append(nodes)

    for i, node_name in enumerate(remove_nodes):
        node = model[node_name[0]]
        output = node.outputs[0]
        next_nodes = model.get_next_nodes(output)
        del_nodes = {node.name}
        for next_node in next_nodes:
            del_nodes.update(find_end(next_node, del_nodes))
        for del_node in del_nodes:
            model.remove(del_node, {})
        if i == 0 or i == 2:
            const = 1/23 * np.ones((bs, 23, 23))
        else:
            const = 1/16 * np.ones((bs, 15, 15))
        const = const.astype('float32')
        name  = 'uniform_' + str(i)
        model.add_initializer(name, const)
        next_node = model.get_next_nodes(model[node_name[1]].outputs[0])[0]
        model.remove(node_name[1], {})
        new_matmul = model.add_node(node_name[1], 'MatMul')
        model.connect_node(new_matmul, [name, node_name[2]], [next_node.name])
    return model

def find_end(node, nodes):
    nodes.add(node.name)
    if node.op_type =='Squeeze':
        return nodes
    for output in node.outputs:
        next_nodes = model.get_next_nodes(output)
        for next_node in next_nodes:
            nodes = find_end(next_node, nodes)
    return nodes

def clear_softmax(model):
    softmax_nodes = model.get_nodes('Softmax')
    for node in softmax_nodes:
        inp = node.inputs[0]
        out = node.outputs[0]
        prev_node = model.get_prev_node(inp)
        if prev_node.op_type != 'Transpose':
            continue
        next_node = model.get_next_nodes(output)[0]
        if next_node.op_type != 'Transpose':
            continue
        axis = node.attrs['axis']
        new_axis = prev_node.attrs['prem'][axis]
        node.attrs['axis'] = new_axis
        model.remove(prev_node.name)
        model.remove(next_node.name)
    return model

def replace_gather(model):
    topk_nodes = model.get_nodes('TopK')
    for i, node in enumerate(topk_nodes):
        f_name = 'Flatten_' + str(i)
        g_name = 'Gather_' + str(i)
        s_name = 'Squeeze_new_' + str(i)
        f_node = model.add_node(f_name, 'Flatten', attrs={'axis': 1})
        g_node = model.add_node(g_name, 'Gather', attrs={'axis': 1})
        s_node = model.add_node(s_name, 'Squeeze', attrs={'axes': [1]})
        e_node = model.get_next_nodes(node.outputs[1])[0]
        ge_node = model.get_next_nodes(e_node.outputs[0])[0]
        f_node.inputs = [node.outputs[1]]
        f_out = 'fatten_out_' + str(i)
        f_node.outputs = [f_node]
        g_node.inputs = [ge_nodes.inputs[0], f_out]
        g_out = 'gather_out_' + str(i)
        g_node.outputs = [g_out]
        s_node.inputs = [g_out]
        s_node.outputs = ge_node.outputs
        model.update_map()
        model.remove(e_node.name, {})
        model.remove(ge_node.name, {})
    return model

def find_cov(model):
    bn_nodes = mdoel.get_nodes('BatchNormalization')
    nodes = (None, None, None, None, None)
    for node in bn_nodes:
        next_node = model.get_next_nodes(node.outputs[0])[0]
        if next_node.op_type == 'Selu':
            for i in range(5):
                nodes[i] = next_node.name
                next_node = model.get_next_nodes(next_node.outputs[0])[0]
            break
        print(node, nodes)
    return nodes

def slice_conv(model):
    conv_1, selu, conv_2, conv_3, add_ = find_cov(model) # 8, 9, 10, 11, 12
    begin_output = model[conv_1].inputs[0]
    attrs1 = model[conv_1].attrs
    attrs2 = model[conv_2].attrs
    attrs3 = model[conv_3].attrs
    input1 = model[conv_1].inputs[1:]
    input2 = model[conv_2].inputs[1:]
    input3 = model[conv_3].inputs[1:]
    final_inputs = model[add_].outputs
    model.remove(conv_1, {})
    model.remove(conv_2, {})
    mdoel.remove(conv_3, {})
    model.remove(selu, {})
    model.remove(add_, {})
    ind = [[[0], [4299]], [[4297], [8597]], [[8595], [12895]], [[12893], [17193]], [[17191], [21490]]]
    model.add_initializer('slice_axis', np.array([-1]))
    concat_all = model.add_node('Concat_new_all', 'Concat', attrs={'axis': -1}, outputs=final_inputs)
    concat_conv = model.add_node('Concat_new_conv', 'Concat', attrs={'axis': -1}, outputs=['after_concat'])
    for i in range(5):
        name_slice_1 = 'Slice1_new_' + str(i)
        name_slice_2 = 'Slice2_new_' + str(i)
        name_1 = 'Conv1_s' + str(i)
        name_2 = 'Conv2_s' + str(i)
        name_3 = 'Conv2_s' + str(i)
        name_selu = 'Selu_new_' + str(i)
        name_add = 'Add_new_' + str(i)
        begin_name = 'ind0_s' + str(i)
        end_name = 'ind1_s' + str(i)
        model.add_initializer(begin_name, np.array(ind[i][0]))
        model.add_initializer(end_name, np.array(ind[i][1]))
        inputs_1 = [begin_output, begin_name, end_name, 'slice_axis']
        inputs_2 = [concat_conv.outputs[0], begin_name, end_name, 'slice_axis']
        outputs_1 = ['sliced1_s' + str(i)]
        outputs_2 = ['after_conv1_s' + str(i)]
        outputs_3 = ['after_selu_s' + str(i)]
        outputs_4 = ['after_conv2_s' + str(i)]
        outputs_5 = ['after_conv3_s' + str(i)]
        outputs_6 = ['sliced2_s' + str(i)]
        input_add = outputs_4 + outputs_5
        output = ['after_add_s' + str(i)]
        attrs_1 = dict()
        attrs_1['kernel_shape'] = attrs1['kernel_shape']
        attrs_2 = dict()
        attrs_2['kernel_shape'] = attrs2['kernel_shape']
        attrs_3 = dict()
        attrs_3['kernel_shape'] = attrs3['kernel_shape']
        if i == 0:
            attrs_1['pads'] = [1, 1, 1, 0]
            attrs_2['pads'] = [0, 1, 0, 0]
            attrs_3['pads'] = [0, 1, 0, 0]
        if i == 4:
            attrs_1['pads'] = [1, 0, 1, 1]
            attrs_2['pads'] = [0, 0, 0, 1]
            attrs_3['pads'] = [0, 0, 0, 1]
        else:
            attrs_1['pads'] = [1, 0, 1, 0]
            attrs_2['pads'] = [0, 0, 0, 0]
            attrs_3['pads'] = [0, 0, 0, 0]
        model.add_node(name_slice_1, 'Slice', inputs=inputs_1, outputs=outputs_1)
        conv_1_in = outputs_1 + input1
        model.add_node(name_1, 'Conv', attrs=attrs_1, inputs=conv_1_in, outputs=outputs_2)
        model.add_node(name_selu, 'Selu', inputs=outputs_2, outputs=outputs_3)
        model.add_node(name_slice_2, 'Slice', inputs=inputs_2, outputs=outputs_6)
        conv_2_in = outputs_6 + input2
        model.add_node(name_2, 'Conv', attrs=attrs_2, inputs=conv_2_in, outputs=outputs_4)
        conv_3_in = outputs_1 + input3
        model.add_node(name_3, 'Conv', attrs=attrs_3, inputs=conv_3_in, outputs=outputs_5)
        model.add_node(name_add, 'Add', inputs=input_add, outputs=output)
        concat_all.inputs += output
        concat_conv.inputs += outputs_3
    model.update_map()
    return model

def conv1d_to_conv2d(model, bs):
    conv_nodes = model.get_nodes('Conv')
    for i, node in enumerate(conv_nodes):
        attrs = node.attrs
        if len(attrs['dilations']) == 1:
            dil = attrs['dilations'][0]
            ks = attrs['kernel_shape'][0]
            pads = attrs['pads']
            stride = attrs['strides'][0]
            attrs['dilations'] = [1, dil]
            attrs['kernel_shape'] = [1, ks]
            attrs['pads'] = [0, pads[0], 0, pads[1]]
            attrs['strides'] = [1, stride]
            name = node.inputs[1]
            weights = model[name].value
            weights = np.expand_dims(weights, axis=-2)
            model[name].value = weights
            prev_unsq = model.get_prev_node(node.inputs[0])
            prev_unsq.attrs['axes'] = [0, 1]
            next_unsq = model.get_next_nodes(node.outputs[0])[0]
            next_abs = model.get_next_nodes(next_unsq.outputs[0])[0]
            next_maxpool = model.get_next_nodes(next_abs.outputs[0])[0]
            next_maxpool.attrs['kernel_shape'] = [1, 3]
            next_maxpool.attrs['strides'] = [1, 3]
            model.remove(next_unsq.name)
            ori_outputs = next_maxpool.Outputs
            new_outputs = ['after_maxpool_' + str(i)]
            next_maxpool.outputs = new_outputs
            reshape = 'Reshape_new_' + str(i)
            model.add_initializer('reshape_shape', np.array([bs, 1, 70, -1]))
            inputs = new_outputs + ['reshape_shape']
            model.add_node(reshape, 'Reshape', inputs=inputs, outputs=ori_outputs)
            maxpool = 'MaxPool_new_' + str(i)
            attrs = {'ceil_mode': 0, 'kernel_shape': [3, 1], 'pads': [0, 0, 0, 0], 'strides': [3, 1]}
            new_maxpool = model.add_node(maxpool, 'MaxPool', attrs=attrs)
            model.insert_node(reshape, new_maxpool)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--input_name', required=True, help='filepath of the original onnx model')
    parser.add_argument('-m2', '--output_name', required=True, help='filepath of the modified onnx model')
    parser.add_argument('-bs', '--batch_size', required=True, help='batch size of the model input')
    parser.add_argument('method', '--modified_method', default='all', help='which method is used to modify. \
                        The options are all, remove, clear, conv, slice, replace. \
                        The replace method can only be used when batch size is 1.')
    args = parser.parse_args()

    input_name = args.input_name
    output_name = args.output_name
    method = args.modified_method
    bs = args.batch_size
    model = OnnxGraph.parse(input_name)
    if method == 'all':
        model = remove_zero_weights(model, bs):
        model = clear_softmax(model)
        if bs == 1:
            model = replace_gather(model)
        model = conv1d_to_conv2d(model, bs)
        model = slice_conv(model)
        elif method == 'remove':
            mdoel = remove_zero_weights(model,bs)
        elif method == 'clear':
            model = clear_softmax(mdoel)
        elif method == 'replace' and bs == 1:
            model = replace_gather(model)
        elif method == 'conv':
            model = conv1d_to_conv2d(model, bs)
        elif method == 'slice':
            model = slice_conv(model)
        else:
            print('No method is applied to the model!')
        model.remove_unused_nodes()
        model.infershape()
        mdoel.save(output_name)
        print('successfully saved the model')