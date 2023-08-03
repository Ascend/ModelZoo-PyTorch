# Copyright 2023 Huawei Technologies Co., Ltd
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
from auto_optimizer import OnnxNode


def keep_dynamic_batch(graph):
    for node in graph.inputs + graph.outputs:
        node.shape[0] = 'batch'
    graph.infershape()


def create_mask(graph, input_node):
    # build mask block: slice->squeeze->equal->cast->mul
    if isinstance(input_node, OnnxNode):
        input_name = input_node.outputs[0]
    else:
        input_name = input_node.name
    slice_start = graph.add_initializer(
        "Slice_start",
        np.array([0], dtype="int64")
    )
    slice_end = graph.add_initializer(
        "Slice_end",
        np.array([1], dtype="int64")
    )
    slice_axis = graph.add_initializer(
        "Slice_axis",
        np.array([-1], dtype="int64")
    )
    slice_node = graph.add_node(
        "Slice_mask",
        "Slice",
        inputs=[input_name, slice_start.name, slice_end.name, slice_axis.name],
        outputs=["out_Slice_mask"]
    )
    squeeze_node = graph.add_node(
        "Squeeze_mask",
        "Squeeze",
        inputs=slice_node.outputs,
        attrs={
            "axes": [-1]
        },
        outputs=["out_Squeeze_mask"]
    )
    equal_init = graph.add_initializer(
        "Equal_value",
        np.array(0, dtype="float32")
    )
    equal_node = graph.add_node(
        "Equal_mask",
        "Equal",
        inputs=squeeze_node.outputs + [equal_init.name],
        outputs=["out_Equal_mask"]
    )
    cast_node = graph.add_node(
        "Cast_mask",
        "Cast",
        attrs={
            'to': 1
        },
        inputs=equal_node.outputs,
        outputs=["out_Cast_mask"]
    )
    mul_init = graph.add_initializer(
        "Mul_value",
        np.array(-65504, dtype="float32")
    )
    mul_node = graph.add_node(
        "Mul_mask",
        "Mul",
        inputs=cast_node.outputs + [mul_init.name],
        outputs=["out_Mul_mask"]
    )
    return mul_node
