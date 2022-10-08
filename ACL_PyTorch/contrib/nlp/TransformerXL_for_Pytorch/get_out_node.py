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
out_names = [n.name for n in model.graph.output]

rstr = ''
for nn in out_names:
    for n in model.graph.node:
        if nn in n.output:
            rstr += n.name + ':0;'
print(rstr[:-1])

