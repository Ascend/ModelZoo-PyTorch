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

import os
from OXInterface import OXDataType, OXGraph, OXInitializer, OXNode

# 清空终端
os.system('clear')

# 加载模型
oxgraph = OXGraph('../models/ctdet_coco_dla_2x.onnx')

# 给节点添加名字
for idx, node in enumerate(oxgraph._node):
    node.name = node.op_type + '_' + str(idx)
    node.doc_string = ''
oxgraph.save_new_model('../models/ctdet_coco_dla_2x_modify.onnx')

# 修改DeformableConv2D的属性
oxgraph = OXGraph('../models/ctdet_coco_dla_2x_modify.onnx')
oxnodes = oxgraph.get_oxnode_by_op_type('DeformableConv2D')
for oxnode in oxnodes:
    oxnode.set_attribute('dilations', [1, 1])
    oxnode.set_attribute('pads', [1, 1])
    oxnode.set_attribute('strides', [1, 1])

# 保存新模型
oxgraph.save_new_model('../models/ctdet_coco_dla_2x_modify.onnx')
