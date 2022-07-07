# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
import sys

def remove(model_name, output_name):
    model = onnx.load(model_name)
    indexAve =0
    indexpad = 0
    indexReluoutput = 0
    for i in range(len(model.graph.node)):
        if model.graph.node[i].op_type == "AveragePool":
            indexAve = i
            print("AveragePool", i, model.graph.node[i].input, model.graph.node[i].output)
        if model.graph.node[i].op_type == "Pad":
            indexpad = i
            indexReluoutput = model.graph.node[i].input[0]


    model.graph.node[indexAve].input[0] = indexReluoutput

    node_list = ["Pad_"+str(indexpad)]
    max_idx = len(model.graph.node)
    rm_cnt = 0
    for i in range(len(model.graph.node)):
        if i < max_idx:
            n = model.graph.node[i - rm_cnt]
            if n.name in node_list:
                print("remove {} total {}".format(n.name, len(model.graph.node)))
                model.graph.node.remove(n)
                max_idx -= 1
                rm_cnt += 1
    onnx.checker.check_model(model)
    onnx.save(model, model_name, output_name)

if __name__ == "__main__":
    model_name = sys.argv[1]
    output_name = sys.argv[2]
    remove(model_name, output_name)