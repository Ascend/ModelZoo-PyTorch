# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
from gener_core.mod_modify.onnx_graph import OXGraph
from gener_core.mod_modify.onnx_node import OXNode
from gener_core.mod_modify.interface import AttrType as AT

mod = OXGraph("decoder.onnx")
Expand_lists = mod.get_nodes_by_optype("Expand")
for i in range(len(Expand_lists)):
    now_expand = mod.get_node(Expand_lists[i])
    cast_node = mod.add_new_node(now_expand.name + "_cast", "Cast",
                                  {"to": (AT.INT, 6)
                                   })
    Expand_first_input_now = mod.get_node(now_expand.input_name[0])
    now_expand.set_input_node(0, [cast_node])
    cast_node.set_input_node(0, [Expand_first_input_now])

Less_lists = mod.get_nodes_by_optype("Less")
for i in range(len(Less_lists)):
    now_expand = mod.get_node(Less_lists[i])
    cast_node = mod.add_new_node(now_expand.name + "_cast", "Cast",
                                  {"to": (AT.INT, 6)
                                   })
    Expand_second_input_now = mod.get_node(now_expand.input_name[1])
    now_expand.set_input_node(1, [cast_node])
    cast_node.set_input_node(0, [Expand_second_input_now])

Greater_lists = mod.get_nodes_by_optype("Greater")
for greater_node in Greater_lists:
    now_expand = mod.get_node(greater_node)
    cast_node = mod.add_new_node(now_expand.name + "_cast", "Cast",
                                  {"to": (AT.INT, 6)
                                   })
    Expand_second_input_now = mod.get_node(now_expand.input_name[1])
    now_expand.set_input_node(1, [cast_node])
    cast_node.set_input_node(0, [Expand_second_input_now])

not_change_cast = []
Range_lists = mod.get_nodes_by_optype("Range")
for range_node in Range_lists:
    now_expand = mod.get_node(range_node)
    Expand_first_input_now = mod.get_node(now_expand.input_name[1])
    not_change_cast.append(Expand_first_input_now.name)

to = 6
Cast = mod.get_nodes_by_optype("Cast")
for cast_node in Cast:
    now_Cast = mod.get_node(cast_node)
    if now_Cast.get_attr("to", AT.INT) == 7 and now_Cast.name not in not_change_cast:
        now_Cast.set_attr({"to": (AT.INT, to)})

Equal = mod.get_nodes_by_optype("Equal")
for equal_node in Equal:
    now_equal = mod.get_node(equal_node)
    now_ends = mod.get_node(now_equal.input_name[1])
    if now_ends.op_type in ("Initializer", "Constant") and now_ends.const_value.dtype == "int64":
        print("now_ends.dtype:", now_ends.const_value.dtype)
        val = now_ends.const_value.astype("int32")
        now_ends.set_const_value(val)

mod.save_new_model("decoder_final.onnx")

