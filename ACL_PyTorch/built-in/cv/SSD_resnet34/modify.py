# Copyright 2023 Huawei Technologies Co., Ltd
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
from auto_optimizer import OnnxGraph

inp = sys.argv[1]
out = sys.argv[2]

g = OnnxGraph.parse(inp)

g.remove('Shape_707')
g.remove('Gather_709')
g.remove('Unsqueeze_710')
g.remove('Concat_712')
g.remove('Cast_713')
g.remove('ReduceMin_714')
g.remove('Cast_715')
g.remove('Unsqueeze_716')

topk = g['TopK_717']
topk.inputs[1] = 'Constant_711'
g.save(out)