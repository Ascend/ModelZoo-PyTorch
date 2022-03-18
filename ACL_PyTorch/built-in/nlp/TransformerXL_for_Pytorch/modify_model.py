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
#!/usr/bin/env python3.7
import sys
import numpy as np
import onnx
from gener_core.mod_modify.interface import AttrType as AT
from gener_core.mod_modify.onnx_graph import OXGraph


def get_node_by_name(graph, node_name):
    for x in graph.node:
        if x.name == node_name:
            return x
    raise RuntimeError("node not found, check node name")


def preprocess():
    model_path = "model_sim.onnx"
    model = onnx.load(model_path)

    remove_node_list = ['Concat_3399', 'Concat_3404', 'Concat_3409', 'Concat_3414', 'Concat_3419', 'Concat_3424',
                        'Concat_3429', 'Concat_3434', 'Concat_3439', 'Concat_3444', 'Concat_3449', 'Concat_3454']
    for n in remove_node_list:
        print("remove {}".format(n))
        model.graph.node.remove(get_node_by_name(model.graph, n))

    """
    合并Concat
    比如
      2      data        2    data
      |      Mul    =>    |   Mul
      |  /  \  |           \  /
    Concat  Concat        Concat
    ...
    """
    get_node_by_name(model.graph, "Slice_3403").input[0] = '273'
    get_node_by_name(model.graph, "Slice_3408").input[0] = '582'
    get_node_by_name(model.graph, "Slice_3413").input[0] = '891'
    get_node_by_name(model.graph, "Slice_3418").input[0] = '1200'
    get_node_by_name(model.graph, "Slice_3423").input[0] = '1509'
    get_node_by_name(model.graph, "Slice_3428").input[0] = '1818'
    get_node_by_name(model.graph, "Slice_3433").input[0] = '2127'
    get_node_by_name(model.graph, "Slice_3438").input[0] = '2436'
    get_node_by_name(model.graph, "Slice_3438").input[0] = '2436'
    get_node_by_name(model.graph, "Slice_3443").input[0] = '2745'
    get_node_by_name(model.graph, "Slice_3448").input[0] = '3054'
    get_node_by_name(model.graph, "Slice_3453").input[0] = '3363'
    get_node_by_name(model.graph, "Slice_3458").input[0] = '3672'

    onnx.save(model, "model_sim_new.onnx")


def make_model(dummy_mod, dst_path):
    mod = OXGraph(dummy_mod)
    io_map = mod.get_net_in_out_map()

    add_nodes = mod.get_nodes_by_optype("Add")
    for add_n in add_nodes:
        mm_node = mod.get_node(add_n.input_name[0])
        if mm_node.op_type == "MatMul":
            weight_node = mod.get_node(mm_node.input_name[1])
            bias_node = mod.get_node(add_n.input_name[1])
            after_add_node = mod.get_node(io_map.get(add_n.name)[0])

            # new a matmul_v2 node
            mmv2_node = mod.add_new_node(f"Gemm_{add_n.input_name[0]}", "Gemm")
            if after_add_node.op_type == "Relu":
                squeeze_node = mod.add_new_node(f"Squeeze_{mm_node.input_name[0]}", "Squeeze",
                                                {"axes": (AT.LIST_INT, [1])})

                squeeze_node.set_input_node(0, [mod.get_node(mm_node.input_name[0])])
                mmv2_node.set_input_node(0, [squeeze_node, weight_node, bias_node])
                after_add_node.set_input_node(0, [mmv2_node])
            elif after_add_node.op_type == "Add":
                unsqueeze_node = mod.add_new_node(f"Unsqueeze_{mm_node.input_name[0]}", "Unsqueeze",
                                                  {"axes": (AT.LIST_INT, [1])})

                mmv2_node.set_input_node(0, [mod.get_node(mm_node.input_name[0]), weight_node, bias_node])
                unsqueeze_node.set_input_node(0, [mmv2_node])
                after_add_node.set_input_node(1, [unsqueeze_node])

            mod.node_remove([mm_node, add_n])

    mm_nodes = mod.get_nodes_by_optype("MatMul")
    for mm_node in mm_nodes:
        w_node = mod.get_node(mm_node.input_name[1])
        if w_node.op_type in ["Constant", "Initializer"]:
            w_value = w_node.const_value
            if len(w_value.shape) == 2:
                squeeze_node = mod.add_new_node(f"Squeeze_{mm_node.input_name[0]}", "Squeeze",
                                                {"axes": (AT.LIST_INT, [1])})
                unsqueeze_node = mod.add_new_node(f"Unsqueeze_{mm_node.input_name[0]}", "Unsqueeze",
                                                  {"axes": (AT.LIST_INT, [1])})
                squeeze_node.set_input_node(0, [mod.get_node(mm_node.input_name[0])])

                mm_node.set_input_node(0, [squeeze_node])
                unsqueeze_node.set_input_node(0, [mm_node])

                after_mm_node = mod.get_node(io_map.get(mm_node.name)[0])
                if after_mm_node.op_type == "Add":
                    after_mm_node.set_input_node(1, [unsqueeze_node])
                else:
                    after_mm_node.set_input_node(0, [unsqueeze_node])
                mm_node.set_op("Gemm")

    mod.save_new_model(dst_path)


if __name__ == "__main__":
    dummy_mod = "model_sim_new.onnx"
    dst_path = f"model_sim_new_modify.onnx"

    preprocess()
    make_model(dummy_mod, dst_path)

