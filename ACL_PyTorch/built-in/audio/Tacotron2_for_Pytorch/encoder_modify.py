import sys
# ============================================================================
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

import onnx
import copy
from onnx import helper, optimizer
from onnx import numpy_helper
import numpy as np

input_model=sys.argv[1]
output_model=sys.argv[2]
model = onnx.load(input_model)
onnx.checker.check_model(model)
model_nodes = model.graph.node

def getNodeByName(nodes, name: str):
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

lstm_w = []
lstm_r = []
lstm_b = []

############################  modify start  #############################
#########LSTM input   W       R      B
lstm_weight_list = ['299', '300', '298']

batch_size = 4
lstm_modify_list = [
                    'LSTM_30',          # lstm_node_name
                    'Transpose_31',     # lstm_output_node_name
                    '101',              # lstm_node_input_X_id
                    '241',              # Transpose_node_output_id
                    '102'               # lstm_node_input_sequence_lens_id
                    ]

## delete LSTM input initial_h and initial_c
remove_reshape_list = [
                        'Shape_16',  'Gather_18',  'Unsqueeze_19',  'Concat_21',  'Expand_22',
                        'Shape_23',  'Gather_25',  'Unsqueeze_26',  'Concat_28',  'Expand_29'
                      ]
############################  modify end  #############################


INTIALIZERS  = model.graph.initializer
for initializer in INTIALIZERS:
    W = numpy_helper.to_array(initializer)
    if initializer.name == lstm_weight_list[0]:
        lstm_w = copy.deepcopy(W)
    if initializer.name == lstm_weight_list[1]:
        lstm_r = copy.deepcopy(W)
    if initializer.name == lstm_weight_list[2]:
        lstm_b = copy.deepcopy(W)



'''
插入slice节点的意义：
反向LSTM算子需要先把数据反向，反向后的数据输入到正向LSTM算子（权重和正向不同），LSTM算子输出再通过 slice节点反向
'''
starts = onnx.helper.make_tensor('starts', onnx.TensorProto.INT64, [1], [-1])
ends = onnx.helper.make_tensor('ends', onnx.TensorProto.INT64, [1], [-sys.maxsize])
axes = onnx.helper.make_tensor('axes', onnx.TensorProto.INT64, [1], [0])
steps = onnx.helper.make_tensor('steps', onnx.TensorProto.INT64, [1], [-1])
model.graph.initializer.append(starts)
model.graph.initializer.append(ends)
model.graph.initializer.append(axes)
model.graph.initializer.append(steps)


'''
读取权重文件中的前半部分，双向LSTM中的W shape=[2,768,2560]，拆分后的W shape= [1,768,2560]。权重文件拆分为两部分： 前部分：lstm_w[:1,:,:]，
前部分权重传递给正向给LSTM（lstm_w_f）
后部分：lstm_w[1:,:,:]，后部分权重传递给反向给LSTM（lstm_w_r）

make_tensor要求传入1维向量，通过flatten转为1维， np.array(lstm_w[:1,:,:]).flatten()


其他修改参考W
'''

lstm_w_f = onnx.helper.make_tensor('lstm_w_f', onnx.TensorProto.FLOAT, [1, 1024, 512], np.array(lstm_w[:1,:,:]).flatten())
lstm_r_f = onnx.helper.make_tensor('lstm_r_f', onnx.TensorProto.FLOAT, [1, 1024, 256], np.array(lstm_r[:1,:,:]).flatten())
lstm_b_f = onnx.helper.make_tensor('lstm_b_f', onnx.TensorProto.FLOAT, [1, 2048], np.array(lstm_b[:1,:]).flatten())
model.graph.initializer.append(lstm_w_f)
model.graph.initializer.append(lstm_r_f)
model.graph.initializer.append(lstm_b_f)


lstm_w_r = onnx.helper.make_tensor('lstm_w_r', onnx.TensorProto.FLOAT, [1, 1024, 512], np.array(lstm_w[1:,:,:]).flatten())
lstm_r_r = onnx.helper.make_tensor('lstm_r_r', onnx.TensorProto.FLOAT, [1, 1024, 256], np.array(lstm_r[1:,:,:]).flatten())
lstm_b_r = onnx.helper.make_tensor('lstm_b_r', onnx.TensorProto.FLOAT, [1, 2048], np.array(lstm_b[1:,:]).flatten())
model.graph.initializer.append(lstm_w_r)
model.graph.initializer.append(lstm_r_r)
model.graph.initializer.append(lstm_b_r)


initial_h = onnx.helper.make_tensor('initial_h', onnx.TensorProto.FLOAT, [1, batch_size, 256], np.zeros((1, 4, 256), dtype=float).flatten())
model.graph.initializer.append(initial_h)


newnode2 = onnx.helper.make_node(
    'LSTM',
    name='LSTM_Add_1',
    inputs=[lstm_modify_list[2], 'lstm_w_r','lstm_r_r','lstm_b_r', lstm_modify_list[4], 'initial_h', 'initial_h'], # modify sequence_lens
    direction = 'reverse',
    hidden_size = 256,
    outputs=['lstm_add', 'lstm_add_h', 'lstm_add_c'],
)



newnode4 = onnx.helper.make_node(
    'Concat',
    name='Concat_Add',
    inputs=['lstm_33', 'lstm_add'],
    axis = 1,
    outputs=['concat_add'],
)

newnode5 = onnx.helper.make_node(
    'Transpose',
    name=lstm_modify_list[1],    # modify Transpose_name
    inputs=['concat_add'],
    perm = [0, 2, 1, 3],
    outputs=[lstm_modify_list[3]],    # modify transposed
)

newnode6 = onnx.helper.make_node(
    'LSTM',
    name=lstm_modify_list[0],
    inputs=[lstm_modify_list[2], 'lstm_w_f','lstm_r_f','lstm_b_f', lstm_modify_list[4], 'initial_h', 'initial_h'],  # modify X and sequence_lens
    direction = 'forward',
    hidden_size = 256,
    outputs=['lstm_33', 'lstm_33_h', 'lstm_33_c'],
)


model.graph.node.insert(1121, newnode2)
model.graph.node.insert(1123, newnode4)
model.graph.node.remove(getNodeByName(model_nodes, lstm_modify_list[1]))
model.graph.node.insert(1124, newnode5)
model.graph.node.remove(getNodeByName(model_nodes, lstm_modify_list[0]))
model.graph.node.insert(1125, newnode6)


for modify_node in remove_reshape_list:
    model.graph.node.remove(getNodeByName(model_nodes, modify_node))


print("encoder onnx adapted to ATC")

onnx.save(model, output_model)
