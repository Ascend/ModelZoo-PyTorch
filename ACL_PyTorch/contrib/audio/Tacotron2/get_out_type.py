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
from gener_core.mod_modify.onnx_graph import OXGraph
from gener_core.mod_modify.onnx_node import OXNode
import sys

input_model = sys.argv[1]
mod = OXGraph(input_model)
out = mod.get_net_output_nodes()

rstr2 = ''
for n in out:
    if n.name != "mask_out":
        rstr2 += n.name + ':0' + ':FP16;'
rstr2 = rstr2.split(';')
rstr2 = rstr2[:-3]
rstr3=''
for str in rstr2:
    rstr3 += str + ";"
print(rstr3[:-1])

