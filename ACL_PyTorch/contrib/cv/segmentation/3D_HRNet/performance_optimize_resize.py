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

import sys
from onnx_tools.OXInterface.OXInterface import OXGraph

def fix_resize(oxgraph, node):
    # fix attr
    node.set_attribute(attr_name='mode', attr_value='nearest')

    
def main(model_path, save_path):
    oxgraph = OXGraph(model_path)
    oxnode_name = ["Resize_1376", "Resize_1363", "Resize_1350", "Resize_585"]
    resize_nodes = []
    for i in range(len(oxnode_name)):
        resize_nodes.append(oxgraph.get_oxnode_by_name(oxnode_name=oxnode_name[i]))
    for node in resize_nodes:
        fix_resize(oxgraph, node)
    oxgraph.save_new_model(save_path)

if __name__ == '__main__':
    input_path = sys.argv[1]
    out_path = sys.argv[2]
    main(input_path, out_path)