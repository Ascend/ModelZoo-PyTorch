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

if __name__ == '__main__':
    model = onnx.load(sys.argv[1])
    graph = model.graph
    node = graph.node
    softmax_node_index = []
    del_group = []
    for i in range(len(node)):
        if node[i].op_type == 'Softmax':
            del_group.append((node[i-1], node[i], node[i+1], i))
    for g in del_group:
        new_input = g[0].input
        new_output = g[2].output
        new_name = g[1].name
        new_index = g[3]
        new_node = onnx.helper.make_node("Softmax", new_input, new_output, new_name, axis=1)
        for n in g[:-1]:
            graph.node.remove(n)
        graph.node.insert(new_index, new_node)
    onnx.save(model, sys.argv[2])