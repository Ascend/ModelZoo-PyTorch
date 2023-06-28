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

import numpy as np
import onnx

from magiconnx import OnnxGraph

model_name = sys.argv[1]
graph = OnnxGraph(model_name)

axes = onnx.helper.make_attribute("axes", [0,1])
rd_min = graph.get_nodes("ReduceMin")[0]
rd_min._node.attribute.append(axes)
rd_max = graph.get_nodes("ReduceMax")[0]
rd_max._node.attribute.append(axes)

us = graph.add_node('Unsq_1', 'Unsqueeze', {'axes': [2]})
graph.insert_node(graph.get_nodes("Conv")[0].name, us, mode='before')
sq = graph.add_node('Sq_291', 'Squeeze', {'axes': [2]})
graph.insert_node(graph.get_nodes('BatchNormalization')[4].name, sq, mode='after')

convs = graph.get_nodes("Conv")
for conv in convs:
    print(conv.name)
    dil = conv['dilations'][0]
    ks = conv['kernel_shape'][0]
    pds = conv['pads'][0]
    stri = conv['strides'][0]
    conv['dilations'] = [1, dil]
    conv['kernel_shape'] = [1, ks]
    conv['pads'] = [0, pds, 0, pds]
    conv['strides'] = [1, stri]
    conv_w = graph[conv.inputs[1]].value
    conv_w = np.expand_dims(conv_w, axis=-2)
    graph[conv.inputs[1]].value = conv_w

graph.save(model_name)