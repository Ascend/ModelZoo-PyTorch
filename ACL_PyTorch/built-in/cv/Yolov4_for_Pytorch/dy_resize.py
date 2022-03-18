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

import sys
import onnx

model_path = sys.argv[1]
model = onnx.load(model_path)

def RemoveNode(graph, node_list):
    max_idx = len(graph.node)
    rm_cnt = 0
    for i in range(len(graph.node)):
        if i < max_idx:
            n = graph.node[i - rm_cnt]
            if n.name in node_list:
                print("remove {} total {}".format(n.name, len(graph.node)))
                graph.node.remove(n)
                max_idx -= 1
                rm_cnt += 1


def ReplaceScales(ori_list, scales_name):
    n_list = []
    for i, x in enumerate(ori_list):
        if i < 2:
            n_list.append(x)
        if i == 3:
            n_list.append(scales_name)
    return n_list


# 替换Resize节点
for i in range(len(model.graph.node)):
    n = model.graph.node[i]
    if n.op_type == "Resize":
        # print("Resize", i, n.input, n.output)
        model.graph.initializer.append(
            onnx.helper.make_tensor('scales{}'.format(i), onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
        )
        newnode = onnx.helper.make_node(
            'Resize',
            name=n.name,
            inputs=ReplaceScales(n.input, 'scales{}'.format(i)),
            outputs=n.output,
            coordinate_transformation_mode='asymmetric',
            cubic_coeff_a=-0.75,
            mode='nearest',
            nearest_mode='floor'
        )
        model.graph.node.remove(model.graph.node[i])
        model.graph.node.insert(i, newnode)
        print("replace {} index {}".format(n.name, i))

node_list = ['Constant_471', 'Constant_430']
RemoveNode(model.graph, node_list)

onnx.save(model, sys.argv[1].split('.')[0] + "_dbs.onnx")
