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
from onnx import numpy_helper as npy_h
from onnx.numpy_helper import to_array as to_arr

onnx_model = onnx.load(sys.argv[1])
graph = onnx_model.graph
node = graph.node

prelu_slope_input = []
init_maps = {}

for i in range(len(node)):
    node_rise = node[i]
    if node_rise.op_type == 'PRelu':
	    prelu_slope_input.append(node_rise.input[1])

for init in graph.initializer:
    init_maps[init.name] = init

for slope in prelu_slope_input:
    init = init_maps[slope]
    new_init = onnx.helper.make_tensor(init.name, onnx.TensorProto.FLOAT, [1], to_arr(init).flatten())
    graph.initializer.remove(init)
    graph.initializer.extend([new_init])

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, sys.argv[2])