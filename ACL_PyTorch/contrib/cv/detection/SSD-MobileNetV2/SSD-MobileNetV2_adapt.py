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

input_model=sys.argv[1]
output_model=sys.argv[2]
model=onnx.load(input_model)
onnx.checker.check_model(model)
model_nodes=model.graph.node

def getNodeByName(nodes, name: str):
    for n in nodes:
        if n.name == name:
            return n
    return -1

Transpose_before = onnx.helper.make_node(
    'Transpose',
    name='Transpose_before_softmax',
    inputs=['940'],
    outputs=['before_softmax'],
    perm=[2,0,1]
)

soft_max_new = onnx.helper.make_node(
    'Softmax',
    name='Softmax_415',
    inputs=['before_softmax'],
    outputs=['after_softmax'],
    axis=0
)

Transpose_after = onnx.helper.make_node(
    'Transpose',
    name='Transpose_after_softmax',
    inputs=["after_softmax"],
    outputs=['scores'],
    perm=[1,2,0]
)

model_nodes.remove(getNodeByName(model_nodes, 'Softmax_415'))
model_nodes.append(Transpose_before)
model_nodes.append(soft_max_new)
model_nodes.append(Transpose_after)

print("mb2-v2-ssd softmax node adapted")
onnx.save(model, output_model)