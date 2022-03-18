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

import sys
import onnx
import copy
from onnx import helper, optimizer
from onnx import numpy_helper, TensorProto
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


############################  modify start  #############################
modify_list = [
                'Greater_5', '109',
                'Greater_28', '133'
              ]
remove_lists =   [
                'Cast_3', 'Greater_5', 'RandomUniform_2',
                'Cast_26', 'Greater_28', 'RandomUniform_25'
                ]
############################  modify end  #############################


random_input_1 = helper.make_tensor_value_info('random1', TensorProto.FLOAT, [256])
random_input_2 = helper.make_tensor_value_info('random2', TensorProto.FLOAT, [256])
model.graph.input.append(random_input_1)
model.graph.input.append(random_input_2)

great = onnx.helper.make_tensor('great', onnx.TensorProto.FLOAT, [], [0.5])
model.graph.initializer.append(great)

newnode1 = onnx.helper.make_node(
    'Greater',
    name=modify_list[0],
    inputs=['random1', 'great'],
    outputs=[modify_list[1]],
)

newnode2 = onnx.helper.make_node(
    'Greater',
    name=modify_list[2],
    inputs=['random2', 'great'],
    outputs=[modify_list[3]],
)

for remove_list in remove_lists:
    model.graph.node.remove(getNodeByName(model_nodes, remove_list))

model.graph.node.insert(1000, newnode1)
model.graph.node.insert(1001, newnode2)

print("tacotron onnx adapted to ATC")

onnx.save(model, output_model)
