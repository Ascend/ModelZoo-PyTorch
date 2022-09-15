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

#coding=utf-8
#torch.__version >= 1.3.0
from magiconnx import OnnxGraph

graph = OnnxGraph('dsfd.onnx')

Resize_543 = graph['Resize_543']
Resize_543['mode'] = 'nearest'
Resize_567 = graph['Resize_567']
Resize_567['mode'] = 'nearest'
Resize_591 = graph['Resize_591']
Resize_591['mode'] = 'nearest'
Resize_615 = graph['Resize_615']
Resize_615['mode'] = 'nearest'
Resize_639 = graph['Resize_639']
Resize_639['mode'] = 'nearest'

graph.save('test.onnx')