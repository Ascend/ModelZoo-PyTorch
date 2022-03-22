# Copyright 2022 Huawei Technologies Co., Ltd
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
from gener_core.mod_modify.onnx_graph import OXGraph
from gener_core.mod_modify.interface import AttrType as AT


FUS_NODE_TRANS = "Transpose"
FUS_NODE_BMM = "MatMul"
input_model = sys.argv[1]
output_model = sys.argv[2]

mod = OXGraph(input_model)
io_map = mod.get_net_in_out_map()
trans_nodes = mod.get_nodes_by_optype(FUS_NODE_TRANS)

for trans_node in trans_nodes:
    if trans_node.get_attr("perm", AT.LIST_INT) == [0, 2, 3, 1]:
        trans_node.set_attr({"perm": (AT.LIST_INT, [0, 2, 1, 3])})
        bmm = io_map.get(trans_node.name)
        if FUS_NODE_BMM in bmm[0]:
            new_bmm = mod.get_node(bmm[0])
            new_bmm.set_attr({"transB": (AT.INT, 1)})

mod.save_new_model(output_model)
print("OK")