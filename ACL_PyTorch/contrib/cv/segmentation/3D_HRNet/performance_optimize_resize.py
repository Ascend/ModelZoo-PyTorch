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

import sys
from magiconnx import OnnxGraph


NODES = ["Resize_1376", "Resize_1363", "Resize_1350", "Resize_585"]


def fix_resize(graph):
    for resize_node in graph.get_nodes(op_type="Resize"):
        if resize_node.name in NODES:
            resize_node["mode"] = 'nearest'


if __name__ == '__main__':
    onnx_graph = OnnxGraph(sys.argv[1])
    fix_resize(onnx_graph)
    onnx_graph.save(sys.argv[2])
