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
import numpy as np

from auto_optimizer import OnnxGraph


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("usage: python3 modify_onnx.py [in_onnx] [out_onnx]")
        
    in_onnx = sys.argv[1]
    out_onnx = sys.argv[2]
    
    graph = OnnxGraph.parse(in_onnx)
    resize_list = graph.get_nodes('Resize')
    for node in resize_list:
        print("[info]modify node: {}".format(node.name))
        node['coordinate_transformation_mode'] = 'pytorch_half_pixel'
        node['cubic_coeff_a'] = -0.75
        node['mode'] = 'linear'
        node['nearest_mode'] = 'floor'
        graph[node.inputs[1]].value = np.array([], dtype=np.float32)
    graph = graph.simplify()
    graph.save(out_onnx)
    print("[info]modify onnx success! onnx saved to : {}".format(out_onnx))
