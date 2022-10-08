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
import sys


def GetNodeIndex(graph, node_name):
    index = 0
    for i in range(len(graph.node)):
        if graph.node[i].name == node_name:
            index = i
            break
    return index

input_dir = sys.argv[1]
output_dir = sys.argv[2]

model = onnx.load(input_dir)
model.graph.node[GetNodeIndex(model.graph,'Concat_291')].input[1] = '390'
node_list = ["Pad_290"]
max_idx = len(model.graph.node)
rm_cnt = 0
for i in range(len(model.graph.node)):
    if i < max_idx:
        n = model.graph.node[i - rm_cnt]
        if n.name in node_list:
            print("remove {} total {}".format(n.name, len(model.graph.node)))
            model.graph.node.remove(n)
            max_idx -= 1
            rm_cnt += 1
            
model.graph.node[GetNodeIndex(model.graph,'Concat_223')].input[1] = '317'
node_list = ["Pad_222"]
max_idx = len(model.graph.node)
rm_cnt = 0
for i in range(len(model.graph.node)):
    if i < max_idx:
        n = model.graph.node[i - rm_cnt]
        if n.name in node_list:
            print("remove {} total {}".format(n.name, len(model.graph.node)))
            model.graph.node.remove(n)
            max_idx -= 1
            rm_cnt += 1            
            

onnx.checker.check_model(model)
onnx.save(model, output_dir)
