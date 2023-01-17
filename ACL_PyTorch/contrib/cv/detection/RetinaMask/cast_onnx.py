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

import os
import onnx
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, default="../weights/npu_8P_model_0020001_bs1_sim.onnx")
    parser.add_argument("--save_dir", type=str, default="../weights")
    args = parser.parse_args()

    onnx_model = onnx.load(args.weight_path)
    graph = onnx_model.graph
    node = graph.node
    # search Concat_742 node id
    for i in range(len(node)):
        if node[i].name == 'Concat_742':
            node_rise = node[i]
            print(node_rise)
            print(i)
    # add new node, cast to float16
    new_scale_node = onnx.helper.make_node(
        "Cast",
        inputs=["bboxs"],
        outputs=['Cast_bboxs'],
        name="Cast_bboxs",
        to=getattr(onnx.TensorProto, "FLOAT16")
    )

    new_scale_node2 = onnx.helper.make_node(
        "Cast",
        inputs=["1702"],
        outputs=['Cast_742'],
        name="Cast_742",
        to=getattr(onnx.TensorProto, "FLOAT16")
    )

    graph.node.insert(400, new_scale_node)
    graph.node.insert(400, new_scale_node2)
    node[402].input[0] = "Cast_742"
    node[402].input[1] = "Cast_bboxs"


    onnx.save(onnx_model, args.save_dir)
