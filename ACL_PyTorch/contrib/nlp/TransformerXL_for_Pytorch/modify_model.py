# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#
#!/usr/bin/env python3.8
import sys
import numpy as np
import onnx
import onnxruntime as rt
from onnx import shape_inference
from onnx import helper, optimizer

model_path = "model_sim.onnx"
model = onnx.load(model_path)

def GetNode(graph, node_name):
    for x in graph.node:
        if x.name == node_name:
            return x
    raise RuntimeError("node not found, check node name")

remove_node_list = ['Concat_3399', 'Concat_3404', 'Concat_3409', 'Concat_3414', 'Concat_3419', 'Concat_3424',
                    'Concat_3429', 'Concat_3434', 'Concat_3439', 'Concat_3444', 'Concat_3449', 'Concat_3454']
for n in remove_node_list:
    print("remove {}".format(n))
    model.graph.node.remove(GetNode(model.graph, n))

"""
合并Concat
比如
  2      data        2    data
  |      Mul    =>    |   Mul
  |  /  \  |           \  /
Concat  Concat        Concat
...
"""
GetNode(model.graph, "Slice_3403").input[0] = '273'
GetNode(model.graph, "Slice_3408").input[0] = '582'
GetNode(model.graph, "Slice_3413").input[0] = '891'
GetNode(model.graph, "Slice_3418").input[0] = '1200'
GetNode(model.graph, "Slice_3423").input[0] = '1509'
GetNode(model.graph, "Slice_3428").input[0] = '1818'
GetNode(model.graph, "Slice_3433").input[0] = '2127'
GetNode(model.graph, "Slice_3438").input[0] = '2436'
GetNode(model.graph, "Slice_3438").input[0] = '2436'
GetNode(model.graph, "Slice_3443").input[0] = '2745'
GetNode(model.graph, "Slice_3448").input[0] = '3054'
GetNode(model.graph, "Slice_3453").input[0] = '3363'
GetNode(model.graph, "Slice_3458").input[0] = '3672'


# eliminate extra nodes
# passes = ['eliminate_deadend', 'eliminate_unused_initializer']
# optimized_model = optimizer.optimize(model, passes)

# onnx.checker.check_model(model)
onnx.save(model, "model_sim_new.onnx")
