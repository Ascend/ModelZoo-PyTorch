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


input_model = sys.argv[1]
output_model = sys.argv[2]

mod = OXGraph(input_model)

transpose_nodes = ['Transpose_60',
    'Transpose_154',
    'Transpose_248',
    'Transpose_342',
    'Transpose_436',
    'Transpose_530',
    'Transpose_624',
    'Transpose_718',
    'Transpose_812',
    'Transpose_906',
    'Transpose_1000',
    'Transpose_1094']
bmm_nodes = ['MatMul_72',
    'MatMul_166',
    'MatMul_260',
    'MatMul_354',
    'MatMul_448',
    'MatMul_542',
    'MatMul_636',
    'MatMul_730',
    'MatMul_824',
    'MatMul_918',
    'MatMul_1012',
    'MatMul_1106']
io_map = mod.get_net_in_out_map()

for transpose_node in transpose_nodes:
    now_trans = mod.get_node(transpose_node)
    now_trans.set_attr({"perm": (AT.LIST_INT, [0, 2, 1, 3])})
for bmm in bmm_nodes:
    now_bmm = mod.get_node(bmm)
    now_bmm.set_attr({"transB": (AT.INT, 1)})

mod.save_new_model(output_model)
print("OK")