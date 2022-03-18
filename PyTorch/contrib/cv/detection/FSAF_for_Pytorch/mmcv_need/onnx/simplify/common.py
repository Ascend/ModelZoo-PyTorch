# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#

import copy
import warnings

import onnx


def add_suffix2name(ori_model, suffix='__', verify=False):
    """Simplily add a suffix to the name of node, which has a numeric name."""
    # check if has special op, which has subgraph.
    special_ops = ('If', 'Loop')
    for node in ori_model.graph.node:
        if node.op_type in special_ops:
            warnings.warn(f'This model has special op: {node.op_type}.')
            return ori_model

    model = copy.deepcopy(ori_model)

    def need_update(name):
        return name.isnumeric()

    def update_name(nodes):
        for node in nodes:
            if need_update(node.name):
                node.name += suffix

    update_name(model.graph.initializer)
    update_name(model.graph.input)
    update_name(model.graph.output)

    for i, node in enumerate(ori_model.graph.node):
        # process input of node
        for j, name in enumerate(node.input):
            if need_update(name):
                model.graph.node[i].input[j] = name + suffix

        # process output of node
        for j, name in enumerate(node.output):
            if need_update(name):
                model.graph.node[i].output[j] = name + suffix
    if verify:
        onnx.checker.check_model(model)

    return model
