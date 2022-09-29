# Copyright 2021 Huawei Technologies Co., Ltd
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
import numpy as np
import onnx
import argparse

# 1:Get batch parameter
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()

onnx_model = onnx.load("./FasterRCNNDCN.onnx")
graph = onnx_model.graph
node = graph.node

node_constant_list = []
for i in range(len(node)):

    if node[i].name == "Constant_29":
        node[i].name = "Constant_29bug"

    if node[i].op_type == 'Constant':
        for attr_id, attr in enumerate(node[i].attribute):
            if attr.t.data_type == 6:
                # 2:Old nodes ready to be deleted
                old_scale_node = node[i]
                data = np.ones((args.batch_size, attr.t.dims[1], attr.t.dims[2], attr.t.dims[3])).astype(np.float32)
                # 3:Prepare new node
                attr.t.dims[0] = args.batch_size
                new_scale_node = onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=node[i].output,
                    value=onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT,
                                                  dims=[attr.t.dims[0], attr.t.dims[1], attr.t.dims[2], attr.t.dims[3]],
                                                  vals=data.tobytes(), raw=True)
                )
                # 4:Create a new node
                graph.node.remove(old_scale_node)  # Delete old node
                graph.node.insert(i, new_scale_node)  # Insert new node


str_bs = str(args.batch_size)
new_model_name = './FasterRCNNDCN_change_bs' + str_bs + '.onnx'
onnx.save(onnx_model, new_model_name)
