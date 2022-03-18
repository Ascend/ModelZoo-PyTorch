# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import onnx
import argparse

from onnx import TensorProto
from onnx import helper,checker

def add_clip_max(args):
    model = onnx.load(args.input_model)
    graph = model.graph

    max_v = helper.make_tensor('max_v', TensorProto.FLOAT, [], [2048.])
    graph.initializer.append(max_v)

    index_list = []
    for i in range(len(graph.node)):
        if graph.node[i].op_type == "Clip":
            clip_node_def = helper.make_node(name=graph.node[i].name,
                                             op_type='Clip',
                                             inputs=[graph.node[i].input[0], graph.node[i].input[1], 'max_v'],
                                             outputs=[graph.node[i].output[0]])
            graph.node.remove(graph.node[i])
            graph.node.insert(i, clip_node_def)

    checker.check_model(model)
    print("add the max initializer of clip")
    onnx.save(model, args.output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", type=str, default="./biggan.onnx",
                        help="input onnx model")
    parser.add_argument("--output-model", type=str, default="./biggan.onnx",
                        help="output onnx model")
    opt = parser.parse_args()

    add_clip_max(opt)
