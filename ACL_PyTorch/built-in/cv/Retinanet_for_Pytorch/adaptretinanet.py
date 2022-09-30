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



import numpy as np
import copy
import os
from gener_core.mod_modify.onnx_graph import OXGraph
from gener_core.mod_modify.onnx_node import OXNode
from gener_core.mod_modify.interface import AttrType as AT
import sclblonnx as so
import onnx

input_path = 'retinanet.onnx'
mod2 = OXGraph(input_path)
NMS = mod2.get_nodes_by_optype("BatchMultiClassNMS")
nodes_before = mod2.get_nodes_forward_node(NMS[0].name, if_self=False)
mod2.node_remove(nodes_before)

o2_p1_1750 = mod2.add_placeholder_node("o2_p1_1750", "float16", [1, 1000, 0, 4])
o2_p2_1753 = mod2.add_placeholder_node("o2_p2_1753", "float16", [1, 1000, 0])
NMS[0].set_input_node(0, [o2_p1_1750, o2_p2_1753])
mod2.save_new_model("o2.onnx")

mod1 = OXGraph(input_path)
nodes_behind = mod1.get_nodes_behind_node(NMS[0].name, if_self=True)
nms_node = mod1.get_node(NMS[0].name)

nms_input1 = mod1.get_node(nms_node.input_name[0])
nms_input2 = mod1.get_node(nms_node.input_name[1])

mod1.mod.graph.output.remove(mod1.mod.graph.output[0])
mod1.mod.graph.output.remove(mod1.mod.graph.output[0])
mod1.mod.graph.output.remove(mod1.mod.graph.output[0])
mod1.mod.graph.output.remove(mod1.mod.graph.output[0])

onnx1 = mod1.node_remove(nodes_behind)
new_1750 = mod1.add_output_node("1750", "float16")
new_1753 = mod1.add_output_node("1753", "float16")
mod1.save_new_model("o1.onnx")

cmd = "amct_onnx calibration --model ./o1.onnx --save_path ./result --input_shape " \
      "\"input0:-1,3,1344,1344\" --data_dir ./int8data --data_types \"float32\""
os.system(cmd)

#change the cast int64 to int32
models = ["retinanet.onnx", "result_deploy_model.onnx", "o2.onnx"]
to = 6
for model in models:
      mod = OXGraph(model)
      Cast = mod.get_nodes_by_optype("Cast")
      for i in range(len(Cast)):
            now_Cast = mod.get_node(Cast[i])
            if now_Cast.get_attr("to", AT.INT) == 7:
                  now_Cast.set_attr({"to":(AT.INT, to)})
      mod.save_new_model(model.split(".")[0] + "_revise.onnx")
# merge the deploy part1 onnx and the part2 onnx

g1 = so.graph_from_file("result_deploy_model_revise.onnx")
g2 = so.graph_from_file("o2_revise.onnx")
g_merge = so.merge(sg1=g1, sg2=g2, io_match=[("1750", "o2_p1_1750"), ("1753", "o2_p2_1753")], complete=False)
so.graph_to_file(g_merge, "retinanet_int8_revise.onnx")

