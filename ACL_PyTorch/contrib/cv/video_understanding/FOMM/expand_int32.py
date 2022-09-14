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
import numpy as np
from magiconnx import OnnxGraph

def insert_cast_node(graph, before_node, node_name, dtype=6):
    cast_node = graph.add_node(
        node_name,
        'Cast',
        {'to': dtype}
    )
    graph.insert_node(before_node, cast_node, mode='after')


def insert_expand_cast_after_shape(graph):
    nodes_list = ['Expand_3124', 'Expand_3094', 'Expand_3104', 'Expand_3114', 'Expand_3282','Expand_2883', 'Expand_3292', 'Expand_3302', 'Expand_2855', 'Expand_2841', 'Expand_2869', 'Expand_3272']
    shape_nodes = graph.get_nodes("Expand")
    for node in shape_nodes:
        node_name = node.name
        #print(node_name)
        if node_name in nodes_list:
            print(node_name)
            insert_name = 'expand_after_{}'.format(node_name)
            insert_cast_node(graph, node_name, insert_name)



if __name__ == '__main__':
    graph = OnnxGraph('taichi-gen-bs1.onnx')
    insert_expand_cast_after_shape(graph)
    #insert_GatherElements_cast_after_shape(graph)
    graph.save('new_expand_taichi_gen_bs1.onnx')