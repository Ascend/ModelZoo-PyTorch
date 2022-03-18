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
sys.path.append("..") 

from gener_core.mod_modify.onnx_graph import OXGraph
from gener_core.mod_modify.onnx_node import OXNode
from gener_core.mod_modify.interface import AttrType as AT

input_model=sys.argv[1]
output_model=sys.argv[2]

mod = OXGraph(input_model)

Softmax_lists = ["Softmax_374"]

for i in range(len(Softmax_lists)):
    modify_node = mod.get_node(Softmax_lists[i])
    cast_node = mod.add_new_node(modify_node.name + "_transpose1", "Transpose",
                                  {"perm": (AT.LIST_INT, [1, 0])
                                   })
    input_node = mod.get_node(modify_node.input_name[0])
    modify_node.set_input_node(0, [cast_node])
    cast_node.set_input_node(0, [input_node])

    io_map = mod.get_net_in_out_map()
    input_node = mod.get_node(input_node.input_name[0])

    modify_node.set_attr({"axis": (AT.INT, 0)})

mod.save_new_model(output_model)
