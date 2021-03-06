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
    Decode = mod2.get_nodes_by_optype("YoloxBoundingBoxDecode")
    nodes_before = mod2.get_nodes_forward_node(Decode[0].name, if_self=False)
    nodes_before.discard('NULL')
    Mul_node = mod2.get_node("Mul_523")
    nodes_before_mul = mod2.get_nodes_forward_node(Mul_node.name, if_self=False)
    nodes_before_mul.discard('NULL')
    remove_nodes = nodes_before_mul.union(nodes_before)
    mod2.node_remove(remove_nodes)

    o2_p1_1750 = mod2.add_placeholder_node("o2_p1_1750", "float32", [1, 1000, 0, 4])
    o2_p2_1753 = mod2.add_placeholder_node("o2_p2_1753", "float32", [1, 1000, 0])
    mul_p3_1113 = mod2.add_placeholder_node("mul_p3_1113", "float32", [1, 1000, 0, 4])
    mul_p4_1119 = mod2.add_placeholder_node("mul_p4_1119", "float32", [1, 1000, 0, 4])
    Decode[0].set_input_node(0, [o2_p1_1750, o2_p2_1753])
    Mul_node.set_input_node(0, [mul_p3_1113, mul_p4_1119])
    mod2.save_new_model("o2.onnx")

    mod1 = OXGraph(input_path)
    nodes_behind = mod1.get_nodes_behind_node(Decode[0].name, if_self=True)
    nms_node = mod1.get_node(Decode[0].name)
    mod1.mod.graph.output.remove(mod1.mod.graph.output[0])
    mod1.mod.graph.output.remove(mod1.mod.graph.output[0])

    nodes_behind_mul = mod1.get_nodes_behind_node(Mul_node.name, if_self=True)
    remove_nodes_behind = nodes_behind.union(nodes_behind_mul)
    onnx1 = mod1.node_remove(remove_nodes_behind)
    new_1750 = mod1.add_output_node("1117", "float32")
    new_1753 = mod1.add_output_node("1116", "float32")
    new_1113 = mod1.add_output_node("1113", "float32")
    new_1119 = mod1.add_output_node("1119", "float32")
    mod1.save_new_model("o1.onnx")

    cmd = "amct_onnx calibration --model ./o1.onnx --save_path ./result --input_shape " \
          "\"input:-1,3,640,640\" --data_dir ./int8data --data_types \"float32\""
    os.system(cmd)

    #change the cast int64 to int32
    int8mod = OXGraph("o2.onnx")
    Expand_lists = ["Add_574", "Add_557"]
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
    int8mod.save_new_model("o2_revise.onnx")
    g1 = so.graph_from_file("result_deploy_model.onnx")
    g2 = so.graph_from_file("o2_revise.onnx")
    g_merge = so.merge(sg1=g1, sg2=g2, io_match=[("1117", "o2_p1_1750"), ("1116", "o2_p2_1753"),
                                                 ("1113", "mul_p3_1113"), ("1119", "mul_p4_1119")], complete=False)
    so.graph_to_file(g_merge, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("yolox tiny generate quant data")
    parser.add_argument('--src_path', type=str, default='./src.onnx', help='src onnx')
    parser.add_argument('--save_path', type=str, default='./res.onnx', help='int8 onnx')
    args = parser.parse_args()
    revise_model(args.src_path, args.save_path)