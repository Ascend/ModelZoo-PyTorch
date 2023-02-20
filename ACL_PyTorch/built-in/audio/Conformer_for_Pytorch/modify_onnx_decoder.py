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

# -*- coding:utf-8 -*-

import numpy as np
import copy
from gener_core.mod_modify.interface import AttrType as AT
from gener_core.mod_modify.onnx_graph import OXGraph

mod = OXGraph("xformer_decoder.onnx")
Gather = mod.get_node("/Gather_2")
Add = mod.get_node("/Slice_6")
Shape_new = mod.add_new_node(Gather.name + "shape", "Shape")
Shape_new.set_input_node(0, [Add])
Gather_new = mod.add_new_node(Gather.name + "gather", "Gather", {"axis":(AT.LIST_INT, [0])})
Gather_new_in2 = mod.add_const_node(Gather_new.name + "input2", np.array(1))
Gather_new.set_input_node(0, [Shape_new, Gather_new_in2])

Sub_new = mod.add_new_node(Gather.name + "sub", "Sub")
Sub_new_in2 = mod.add_const_node(Sub_new.name + "input2", np.array(1))
Sub_new.set_input_node(0, [Gather_new, Sub_new_in2])
Gather.set_input_node(0, [Add, Sub_new])
mod.save_new_model("xformer_decoder_revise.onnx")