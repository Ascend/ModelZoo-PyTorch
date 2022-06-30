# Copyright 2022 Huawei Technologies Co., Ltd
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
import argparse

def revise_model(src_path, save_path):
    input_path = src_path
    mod2 = OXGraph(input_path)
    NMS = mod2.get_nodes_by_optype("BatchMultiClassNMS")
    nodes_before = mod2.get_nodes_forward_node(NMS[0].name, if_self=False)
    nodes_before.discard('NULL')
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

    onnx1 = mod1.node_remove(nodes_behind)
    new_1750 = mod1.add_output_node("1131", "float16")
    new_1753 = mod1.add_output_node("1144", "float16")
    mod1.save_new_model("o1.onnx")

    cmd = "amct_onnx calibration --model ./o1.onnx --save_path ./result --input_shape " \
          "\"input:-1,3,640,640\" --data_dir ./int8data --data_types \"float32\""
    os.system(cmd)

    # change the cast int64 to int32
    int8mod = OXGraph("result_deploy_model.onnx")
    Expand_lists = ["Add_349", "Add_315", "Add_523", "Add_332", "Add_506"]
    for i in range(len(Expand_lists)):
        now_expand = int8mod.get_node(Expand_lists[i])
        cast_node1 = int8mod.add_new_node(now_expand.name + "_cast_1", "Cast",
                                          {"to": (AT.INT, 6)
                                           })
        cast_node2 = int8mod.add_new_node(now_expand.name + "_cast_2", "Cast",
                                          {"to": (AT.INT, 6)
                                           })
        Expand_first_input_now = int8mod.get_node(now_expand.input_name[0])
        Expand_second_input_now = int8mod.get_node(now_expand.input_name[1])
        now_expand.set_input_node(0, [cast_node1, cast_node2])
        cast_node1.set_input_node(0, [Expand_first_input_now])
        cast_node2.set_input_node(0, [Expand_second_input_now])
    int8mod.save_new_model("result_deploy_model_revise.onnx")
    g1 = so.graph_from_file("result_deploy_model_revise.onnx")
    g2 = so.graph_from_file("o2.onnx")
    g_merge = so.merge(sg1=g1, sg2=g2, io_match=[("1131", "o2_p1_1750"), ("1144", "o2_p2_1753")], complete=False)
    so.graph_to_file(g_merge, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("yolof generate quant onnx")
    parser.add_argument('--src_path', type=str, default='./src.onnx', help='src onnx')
    parser.add_argument('--save_path', type=str, default='./res.onnx', help='int8 onnx')
    args = parser.parse_args()
    revise_model(args.src_path, args.save_path)