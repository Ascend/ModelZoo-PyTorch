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

import onnx
import sys
import os

if len(sys.argv) < 3:
    raise Exception("usage: python3 xxx.py [src_path] [save_path]")
in_onnx=sys.argv[1]
out_onnx=sys.argv[2]
in_onnx = os.path.realpath(in_onnx)
out_onnx = os.path.realpath(out_onnx)


onnx_model = onnx.load(in_onnx)
graph = onnx_model.graph
nodes = graph.node

# Resize mode: 'linear' to 'nearest'.
idx = 0
for node in nodes:
    if node.op_type == 'Resize' and node.attribute[2].s == b'linear':
        if idx == 1:
            print("{} mode to 'nearest'.".format(node.name))
            node.attribute[2].s = b'nearest'
        idx += 1
        
# insert a Cast op after ArgMax_204.
for i in range(len(nodes)):
    if nodes[i].op_type == 'ArgMax':
        print("insert Cast after {}.".format(nodes[i].name))
        argmax = nodes[i]
        argmax_id = i

cast_input0 = "cast_in"
cast_output0 = "cast_out"
cast_new = onnx.helper.make_node(
    "Cast",
    inputs=[cast_input0],
    outputs=[cast_output0],
    name="Cast_new",
    to=getattr(onnx.TensorProto,"INT32"))

argmax.output[0] = cast_input0
nodes[argmax_id+1].input[0] = cast_output0

graph.node.insert(argmax_id + 1, cast_new)

onnx.save(onnx_model, out_onnx)
