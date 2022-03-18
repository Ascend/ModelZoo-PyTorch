# Copyright 2021 Huawei Technologies Co., Ltd
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

# -*- coding:utf-8 -*-
import sys
import numpy as np
import onnx
from gener_core.mod_modify.interface import AttrType as AT
from gener_core.mod_modify.onnx_graph import OXGraph


def make_model(dummy_mod, dst_path):
    mod = OXGraph(dummy_mod)
    io_map = mod.get_net_in_out_map()

    gather_nodes = mod.get_nodes_by_optype("Gather")
    for g_node in gather_nodes:
        concat_node = mod.get_node(g_node.input_name[0])
        indices_node = mod.get_node(g_node.input_name[1])
        match = False
        if concat_node.op_type == "Concat" and indices_node.op_type in ["Constant", "Initializer"]:
            match = True
        if match:
            indices = indices_node.const_value
            cat_axis = concat_node.get_attr("axis", AT.INT)
            g_axis = g_node.get_attr("axis", AT.INT)
            if cat_axis == -1 and g_axis == 3:  # 输入shape是4维，这里3与-1等价，Concat和Gather可以消掉
                through_node = mod.get_node(concat_node.input_name[indices])  # Concat对应的输入作为Gather后算子的输入
                squeeze_node = mod.add_new_node(f"Squeeze_{g_node.input_name[0]}", "Squeeze",
                                                {"axes": (AT.LIST_INT, [-1])})
                after_gather_node = mod.get_node(io_map.get(g_node.name)[0])
                squeeze_node.set_input_node(0, [through_node])
                after_gather_node.set_input_node(0, [squeeze_node])  # 添加一个Squeeze算子恢复维度

    # BatchMatMulV2添加bias
    mm_nodes = mod.get_nodes_by_optype("MatMul")
    for mm_node in mm_nodes:
        w_node = mod.get_node(mm_node.input_name[1])
        if w_node.op_type in ["Constant", "Initializer"]:
            w_value = w_node.const_value
            if len(w_value.shape) == 2:
                add_node = mod.get_node(io_map.get(mm_node.name)[0])

                # 后面为Add算子，与MatMul一对一连接，满足bias条件，则融合
                if len(io_map.get(mm_node.name)) == 1 and add_node.op_type == "Add":
                    bias_node = mod.get_node(add_node.input_name[1])  # 获取Add算子第二个输入
                    if bias_node.op_type in ["Constant", "Initializer"]:  # Add算子第二个输入是Const时，才融合
                        BIAS_IDX = 2
                        mm_node.set_input_node(BIAS_IDX, [bias_node])  # 设置MatMul算子的Bias为Add算子第二个输入
                        # 对Add连接多个节点的情况进行处理
                        node_set = io_map.get(add_node.name)
                        after_add_nodes = []
                        for xn in node_set:
                            after_add_nodes.append(mod.get_node(xn))
                        for after_add_node in after_add_nodes:
                            in_nodes = []
                            for xn in after_add_node.input_name:
                                in_nodes.append(mod.get_node(xn).name)
                            add_idx = in_nodes.index(add_node.name)
                            after_add_node.set_input_node(add_idx, [mm_node])  # 设置Add后的算子输入为MatMul
                        mod.node_remove([add_node.name])  # 删除Add算子

    mod.save_new_model(dst_path)


if __name__ == "__main__":
    dummy_mod = sys.argv[1]
    dst_path = os.path.realpath(sys.argv[1]).split('.')[-2] + "_modify.onnx"
    make_model(dummy_mod, dst_path)

