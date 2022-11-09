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
from MagicONNX.magiconnx.graph import OnnxGraph

graph = OnnxGraph('pointnetcnn.onnx')
splits = graph.get_nodes('Split')
for i in splits:
    graph.del_node(i.name)
Concat = graph.get_nodes('Concat')
list_mo = []
list_mu = []
list_mo_index = []
list_mu_in = []
list_mo_all = []
for i in Concat:
    if len(i.inputs)==1:
        list_mo.append(i)
    else:
        list_mu.append(i)
for i in list_mu:
    list_mu_in.extend(i.inputs)
for i in list_mo:
    if i.outputs[0] in list_mu_in:
        list_mo_index.append(list_mo.index(i))
list_mo_all = [i for i in range(len(list_mo))]
list_mo_index0 = [i for i in list_mo_all if i not in list_mo_index]
for i in list_mo_index:
    graph.del_node(list_mo[i].name, maps={0:1})
for i in list_mo_index0:
    graph.del_node(list_mo[i].name)
graph.save('pointnetcnn_modify.onnx')
