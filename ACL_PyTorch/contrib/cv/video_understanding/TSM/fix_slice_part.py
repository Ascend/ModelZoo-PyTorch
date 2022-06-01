"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
"""
import sys
import numpy as np
import onnx
from magiconnx import OnnxGraph


DIMS = [
    (8, 3136), (32, 3136), (32, 3136), (32, 3136),
    (64, 784), (64, 784), (64, 784), (64, 784),
    (128, 196), (128, 196), (128, 196), (128, 196),
    (128, 196), (128, 196), (256, 49), (256, 49)
]


def fix_slice_part(start_idx, dims, batch_size):
    # get node name
    slice_node1_1 = 'Slice_' + str(start_idx)
    slice_node1_2 = 'Slice_' + str(start_idx + 22)
    concat_node1 = 'Concat_' + str(start_idx + 23)
    const_name = 'Constant_'  +str(start_idx)

    slice_node2_1 = 'Slice_' + str(start_idx + 6)
    slice_node2_2 = 'Slice_' + str(start_idx + 34)
    concat_node2 = 'Concat_' + str(start_idx + 35)

    slice_node3_1 = 'Slice_' + str(start_idx +17)
    slice_node3_2 = 'Slice_' + str(start_idx +29)
    sub_node_1 = 'Sub_' + str(start_idx + 12)
    sub_node_2 = 'Sub_' + str(start_idx + 24)

    # merge slice
    def merge_slice(node1_name, node2_name, node3_name, node3_idx=0):
        node1 = onnx_graph[node1_name]
        node2 = onnx_graph[node2_name]
        for idx, input_para1 in enumerate(node1.inputs[1:]):
            input_para2 = node2.inputs[idx + 1]
            merged_value = np.concatenate(
                [onnx_graph[input_para1].value, onnx_graph[input_para2].value], axis=0)
            init_name = node1_name + '_{}'.format(idx)
            onnx_graph.add_initializer(init_name, merged_value)
            node1.inputs[idx + 1] = init_name

        onnx_graph[node3_name].inputs[node3_idx] = onnx_graph[node2_name].inputs[0]
        onnx_graph.del_node(name=node2_name, auto_connection=False)

    merge_slice(slice_node1_1, slice_node1_2, concat_node1)
    merge_slice(slice_node2_1, slice_node2_2, concat_node2, 1)

    # build and replace constant node
    # values = np.random.rand(1, 1, dims[0], dims[1]).astype(np.float32)
    values = np.zeros([batch_size, 1, dims[0], dims[1]]).astype('float32')
    const_node = onnx_graph.add_node(
        const_name,
        'Constant',
        {
            'value': onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float)
            )
        }
    )
    const_node.outputs[0] = const_name + "_out"

    def replace_const(node1_name, node2_name, node3_name, node3_idx=0):
        # del node1 and node2, insert constant node as input of node3
        onnx_graph.del_node(name=node1_name, auto_connection=False)
        onnx_graph.del_node(name=node2_name, auto_connection=False)
        onnx_graph[node3_name].inputs[node3_idx] = onnx_graph[const_name].outputs[0]

    replace_const(sub_node_1, slice_node3_1, concat_node1, 1)
    replace_const(sub_node_2, slice_node3_2, concat_node2)



if __name__ == '__main__':
    onnx_graph = OnnxGraph(sys.argv[1])
    save_path = sys.argv[2]
    batch_size = int(sys.argv[3])
    for idx, start_idx in enumerate([40, 114, 187, 260,
                                     334, 407, 480, 553,
                                     627, 700, 773, 846,
                                     919, 992, 1066, 1139]):
        dims = DIMS[idx]
        fix_slice_part(start_idx, dims, batch_size)
    onnx_graph.save(save_path)
