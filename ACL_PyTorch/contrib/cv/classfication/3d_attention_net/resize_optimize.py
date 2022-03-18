# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import onnx
import numpy as np
from pprint import pprint
from onnx_tools.OXInterface.OXInterface import OXDataType, OXGraph, OXInitializer, OXNode

oxgraph = OXGraph('3d_attention_net.onnx')
oxnode = oxgraph.get_oxnode_by_name('Resize_77')
oxnode.set_attribute(attr_name='coordinate_transformation_mode', attr_value="asymmetric")
oxnode.set_attribute(attr_name='mode', attr_value="nearest")

oxnode = oxgraph.get_oxnode_by_name('Resize_96')
oxnode.set_attribute(attr_name='coordinate_transformation_mode', attr_value="asymmetric")
oxnode.set_attribute(attr_name='mode', attr_value="nearest")

oxnode = oxgraph.get_oxnode_by_name('Resize_173')
oxnode.set_attribute(attr_name='coordinate_transformation_mode', attr_value="asymmetric")
oxnode.set_attribute(attr_name='mode', attr_value="nearest")

oxnode = oxgraph.get_oxnode_by_name('Resize_241')
oxnode.set_attribute(attr_name='coordinate_transformation_mode', attr_value="asymmetric")
oxnode.set_attribute(attr_name='mode', attr_value="nearest")

oxgraph.save_new_model('3d_attention_net_resize_optimized.onnx')