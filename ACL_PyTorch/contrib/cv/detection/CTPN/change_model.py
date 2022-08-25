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
import sys
import argparse
import copy
import numpy as np
import onnx
from onnx import numpy_helper


# 模型转换相应的参数
parser = argparse.ArgumentParser(description='change onnx model one bidirectional GRU to two forward GRU')
parser.add_argument('--input_path', default='ctpn.onnx',
                    type=str, help='input onnx model path')
parser.add_argument('--output_path', default='ctpn_change.onnx',
                    type=str, help='output onnx model path')
args = parser.parse_args()

input_model=args.input_path
output_model=args.output_path
model = onnx.load(input_model)
onnx.checker.check_model(model)
model_nodes = model.graph.node


def getNodeByName(nodes, name):
    for n in nodes:
        if n.name == name:
            return n
    return -1
    
def GetNodeIndex(graph, node_name):
    index = 0
    for i in range(len(graph.node)):
        if graph.node[i].name == node_name:
            index = i
            break
    return index
    
    
INTIALIZERS  = model.graph.initializer
for initializer in INTIALIZERS:
    W = numpy_helper.to_array(initializer)
    if initializer.name == '328':
        gru_58_w = copy.deepcopy(W)
    if initializer.name == '329':
        gru_58_r = copy.deepcopy(W)
    if initializer.name == '330':
        gru_58_b = copy.deepcopy(W)


# 这是GRU前面的slice算子
starts = onnx.helper.make_tensor('starts', onnx.TensorProto.INT64, [1], [-1])
ends = onnx.helper.make_tensor('ends', onnx.TensorProto.INT64, [1], [-sys.maxsize])
axes = onnx.helper.make_tensor('axes', onnx.TensorProto.INT64, [1], [0])
steps = onnx.helper.make_tensor('steps', onnx.TensorProto.INT64, [1], [-1])
model.graph.initializer.append(starts)
model.graph.initializer.append(ends)
model.graph.initializer.append(axes)
model.graph.initializer.append(steps)
# 因为GRU的initial_h是通过上面的输出获得的因此也要通过slice进行拆分获取，前向
starts_f = onnx.helper.make_tensor('starts_f', onnx.TensorProto.INT64, [1], [0])
ends_f = onnx.helper.make_tensor('ends_f', onnx.TensorProto.INT64, [1], [1])
axes_f = onnx.helper.make_tensor('axes_f', onnx.TensorProto.INT64, [1], [0])
model.graph.initializer.append(starts_f)
model.graph.initializer.append(ends_f)
model.graph.initializer.append(axes_f)
# 因为GRU的initial_h是通过上面的输出获得的因此也要通过slice进行拆分获取，反向
starts_b = onnx.helper.make_tensor('starts_b', onnx.TensorProto.INT64, [1], [1])
ends_b = onnx.helper.make_tensor('ends_b', onnx.TensorProto.INT64, [1], [2])
axes_b = onnx.helper.make_tensor('axes_b', onnx.TensorProto.INT64, [1], [0])
model.graph.initializer.append(starts_b)
model.graph.initializer.append(ends_b)
model.graph.initializer.append(axes_b)

gru_new_w_f = onnx.helper.make_tensor('gru_new_w_f', onnx.TensorProto.FLOAT, [1, 384, 512], np.array(gru_58_w[:1,:,:]).flatten())
gru_new_r_f = onnx.helper.make_tensor('gru_new_r_f', onnx.TensorProto.FLOAT, [1, 384, 128], np.array(gru_58_r[:1,:,:]).flatten())
gru_new_b_f = onnx.helper.make_tensor('gru_new_b_f', onnx.TensorProto.FLOAT, [1, 768], np.array(gru_58_b[:1,:]).flatten())
model.graph.initializer.append(gru_new_w_f)
model.graph.initializer.append(gru_new_r_f)
model.graph.initializer.append(gru_new_b_f)
    
# 下面是反向的GRU
gru_new_w_b = onnx.helper.make_tensor('gru_new_w_b', onnx.TensorProto.FLOAT, [1, 384, 512], np.array(gru_58_w[1:,:,:]).flatten())
gru_new_r_b = onnx.helper.make_tensor('gru_new_r_b', onnx.TensorProto.FLOAT, [1, 384, 128], np.array(gru_58_r[1:,:,:]).flatten())
gru_new_b_b = onnx.helper.make_tensor('gru_new_b_b', onnx.TensorProto.FLOAT, [1, 768], np.array(gru_58_b[1:,:]).flatten())
model.graph.initializer.append(gru_new_w_b)
model.graph.initializer.append(gru_new_r_b)
model.graph.initializer.append(gru_new_b_b)

newnode_Slice_f = onnx.helper.make_node(
    'Slice',
    name='Slice_new_f',
    inputs=['103', 'starts_f', 'ends_f','axes_f'],
    outputs=['GRU_new_2_initial_h_input'],
)

newnode_Slice_b = onnx.helper.make_node(
    'Slice',
    name='Slice_new_b',
    inputs=['103', 'starts_b', 'ends_b','axes_b'],
    outputs=['GRU_new_1_initial_h_input'],
)


newnode1 = onnx.helper.make_node(
    'Slice',
    name='Slice_new_1',
    inputs=['104', 'starts', 'ends','axes','steps'],
    outputs=['GRU_new_1_input'],
)
    
newnode2 = onnx.helper.make_node(
    'GRU',
    name='GRU_new_1',
    inputs=['GRU_new_1_input', 'gru_new_w_b','gru_new_r_b','gru_new_b_b', '', 'GRU_new_1_initial_h_input'],
    direction = 'forward',
    hidden_size = 128,
    linear_before_reset = 1,
    outputs=['Slice_new_2_input'],
)
    
newnode3 = onnx.helper.make_node(
    'Slice',
    name='Slice_new_2',
    inputs=['Slice_new_2_input', 'starts', 'ends','axes','steps'],
    outputs=['Concat_new_input_1'],
)

newnode4 = onnx.helper.make_node(
    'GRU',
    name='GRU_new_2',
    inputs=['104', 'gru_new_w_f','gru_new_r_f','gru_new_b_f', '', 'GRU_new_2_initial_h_input'],
    direction = 'forward',
    hidden_size = 128,
    linear_before_reset = 1,
    outputs=['Concat_new_input_2'],
)

newnode5 = onnx.helper.make_node(
    'Concat',
    name='Concat_new',
    inputs=['Concat_new_input_2', 'Concat_new_input_1'],
    axis = 1,
    outputs=['221'],
)


model.graph.node.remove(getNodeByName(model_nodes, 'GRU_58'))
model.graph.node.insert(58, newnode_Slice_f)
model.graph.node.insert(59, newnode1)
model.graph.node.insert(60, newnode2)
model.graph.node.insert(61, newnode3)
model.graph.node.insert(62, newnode_Slice_b)
model.graph.node.insert(63, newnode4)
model.graph.node.insert(64, newnode5)


onnx.save(model, output_model)
print("change onnx model success")