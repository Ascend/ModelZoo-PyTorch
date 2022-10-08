# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import numpy as np
from magiconnx import OnnxGraph

onnx_name = sys.argv[1]
batch_size = sys.argv[2]
onnx_name_bs = sys.argv[3]

bs = int(batch_size)

graph = OnnxGraph(onnx_name)

concat_list = ("Concat_36", "Concat_162", "Concat_342", "Concat_522", "Concat_703", "Concat_883", "Concat_1063",
               "Concat_1243", "Concat_1424", "Concat_1604", "Concat_1784", "Concat_1964")
slice_list = ("Slice_259", "Slice_439", "Slice_619", "Slice_800", "Slice_980", "Slice_1160", "Slice_1340",
              "Slice_1521", "Slice_1701", "Slice_1881", "Slice_2061")
gather_list = ('Gather_70', 'Gather_72', 'Gather_74', 'Gather_197', 'Gather_199', 'Gather_201', 'Gather_377',
               'Gather_379', 'Gather_381', 'Gather_557', 'Gather_559', 'Gather_561', 'Gather_738', 'Gather_740',
               'Gather_742', 'Gather_918', 'Gather_920', 'Gather_922', 'Gather_1098', 'Gather_1100', 'Gather_1102',
               'Gather_1278', 'Gather_1280', 'Gather_1282', 'Gather_1459', 'Gather_1461', 'Gather_1463',
               'Gather_1639', 'Gather_1641', 'Gather_1643', 'Gather_1819', 'Gather_1821', 'Gather_1823',
               'Gather_1999', 'Gather_2001', 'Gather_2003')

pad_zero = np.zeros((1, 11, 384), dtype="float32")
OnnxGraph.add_node(graph, 'expand_m', 'Expand')
OnnxGraph.add_initializer(graph, name='pad_zero', value=pad_zero)

graph['expand_m'].node.input[0] = 'pad_zero'
graph['expand_m'].node.input.append('736')
graph['expand_m'].node.output[0] = 'pad_out'

for name in concat_list:
    graph[name].node.input.append('pad_out')

stop=np.array([197])
num_0 = np.array([0])
num_1 = np.array([1])
num_2 = np.array([2])

OnnxGraph.add_initializer(graph, name='stop', value=stop)
OnnxGraph.add_initializer(graph, name='num0', value=num_0)
OnnxGraph.add_initializer(graph, name='num1', value=num_1)
OnnxGraph.add_initializer(graph, name='num2', value=num_2)
pad_zero2 = np.zeros((4, 6, 11, 64), dtype="float32")
OnnxGraph.add_initializer(graph, name='pad_zero2', value=pad_zero2)

for name in gather_list:
    new_name = 'Concat' + name
    OnnxGraph.add_node(graph, new_name, 'Concat', attrs={"axis": 2})
    graph.insert_node(name, graph[new_name], index=0, mode='after')
    pad_zero2 = np.zeros((bs, 6, 11, 64), dtype="float32") #修改bs大小
    OnnxGraph.add_initializer(graph, name=name+'pad_zero', value=pad_zero2)
    graph[new_name].node.input.append(name+'pad_zero')
    new_name = 'slice'+name
    OnnxGraph.add_node(graph, new_name, 'Slice')
    graph.insert_node(name, graph[new_name], index=0, mode='after')
    graph[new_name].node.input.append('num0')
    graph[new_name].node.input.append('stop')
    graph[new_name].node.input.append('num2')
    graph[new_name].node.input.append('num1')

for name in slice_list:
    graph[name].node.input[2]='stop'

graph.save(onnx_name_bs)
