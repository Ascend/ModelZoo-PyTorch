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
import numpy as np
from auto_optimizer import OnnxGraph


def fix_resize(graph):
    for ind, node in enumerate(graph.get_nodes('Resize')):

        concat_0 = graph.get_prev_node(node.inputs[1])
        div_0 = graph.get_prev_node(concat_0.inputs[1])
        cast_0 = graph.get_prev_node(div_0.inputs[0])
 
        graph.remove(concat_0.name)
        graph.remove(cast_0.name)

        cast_1 = graph.get_prev_node(div_0.inputs[1])
        slice_0 = graph.get_prev_node(cast_1.inputs[0])
        shape_0 = graph.get_prev_node(slice_0.inputs[0])

        graph.remove(shape_0.name)
        graph.remove(slice_0.name)
        graph.remove(cast_1.name)
        graph.remove(div_0.name)

        concat_1 = graph.get_prev_node(cast_0.inputs[0])

        un_0 = graph.get_prev_node(concat_1.inputs[0])
        floor_0 = graph.get_prev_node(un_0.inputs[0])
        cast_2 = graph.get_prev_node(floor_0.inputs[0])
        mul_0 = graph.get_prev_node(cast_2.inputs[0])

        B_value = graph[mul_0.inputs[-1]].value
        B_value = B_value.reshape(1,)
        scale_value = np.concatenate((np.array([1,1]),B_value,B_value)).astype(np.float32)
        
        new_scale = graph.add_initializer(f'{node.name}_sca', scale_value)
        gather_0 = graph.get_prev_node(mul_0.inputs[0])
        shape_1 = graph.get_prev_node(gather_0.inputs[0])

        graph.remove(concat_1.name)
        graph.remove(un_0.name)
        graph.remove(floor_0.name)
        graph.remove(cast_2.name)
        graph.remove(mul_0.name)
        graph.remove(gather_0.name)
        graph.remove(shape_1.name)


        un_1 = graph.get_prev_node(concat_1.inputs[1])
        floor_1 = graph.get_prev_node(un_1.inputs[0])
        cast_3 = graph.get_prev_node(floor_1.inputs[0])
        mul_1 = graph.get_prev_node(cast_3.inputs[0])
        gather_1 = graph.get_prev_node(mul_1.inputs[0])
        shape_2 = graph.get_prev_node(gather_1.inputs[0])

        graph.remove(un_1.name)
        graph.remove(floor_1.name)
        graph.remove(cast_3.name)
        graph.remove(mul_1.name)
        graph.remove(gather_1.name)
        graph.remove(shape_2.name)

        node.inputs[-1] = new_scale.name






if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]

    onnx_graph = OnnxGraph.parse(input_path)
    fix_resize(onnx_graph)
    onnx_graph.save(save_path)