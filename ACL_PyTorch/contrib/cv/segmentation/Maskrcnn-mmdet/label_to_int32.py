# Copyright 2020 Huawei Technologies Co., Ltd
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

import onnx

def getNodeByName(nodes, name: str):
    for n in nodes:
        if n.name == name:
            return n
    
    return -1

model = onnx.load("maskrcnn_r50_fpn_1x.onnx")
cast = getNodeByName(model.graph.node, 'Cast_622')
cast.attribute[0].i = 6
onnx.save(model, 'maskrcnn_r50_fpn_1x.onnx')