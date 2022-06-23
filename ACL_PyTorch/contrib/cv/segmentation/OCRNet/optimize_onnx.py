# Copyright 2022 Huawei Technologies Co., Ltd
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
in_onnx = sys.argv[1]
out_onnx = sys.argv[2]

onnx_model = onnx.load(in_onnx)
graph = onnx_model.graph
nodes = graph.node

optimize_resize_list = ['Resize_0', 'Resize_2', 'Resize_3', 'Resize_5', 'Resize_7', 'Resize_9', 'Resize_14', 
                        'Resize_16', 'Resize_18', 'Resize_19', 'Resize_26', 'Resize_28', 'Resize_29', 'Resize_32']

for node in nodes:
    if node.name in optimize_resize_list:
        node.attribute[1].s = b'nearest'

onnx.save(onnx_model, out_onnx)
