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
import sys
import onnx

model = onnx.load(sys.argv[1])
in_names = [n.name for n in model.graph.input]
out_names = [n.name for n in model.graph.output]

in_names_str = ''
for in_name in in_names:
    if in_name == 'data':
        continue
    in_names_str += in_name + ';'
print(in_names_str[:-1], end='end')


out_names_str = ''
for nn in out_names:
    for n in model.graph.node:
        if nn in n.output:
            out_names_str += n.name + ':0;'
print(out_names_str[:-1], end='')
