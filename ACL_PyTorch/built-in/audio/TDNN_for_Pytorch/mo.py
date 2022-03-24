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

from magiconnx import OnnxGraph

bs = sys.argv[1]
model_name = 'tdnn_bs%s'%bs
graph = OnnxGraph(model_name+'.onnx')
ph = graph.add_placeholder('random','float32',[64,1500])

rm = graph.get_nodes("ReduceMin")[0]
rm.inputs = ['random']
sub = graph.get_nodes("Sub")[-1]
sub.inputs = ['random', rm.outputs[0]]

rn = graph.get_nodes("RandomNormalLike")[0]
graph.del_node(rn.name, auto_connection=False)

graph.save('%s_mod.onnx'%model_name)