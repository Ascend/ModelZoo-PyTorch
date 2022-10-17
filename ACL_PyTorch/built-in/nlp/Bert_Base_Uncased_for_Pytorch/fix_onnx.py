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


import sys
import numpy as np
from magiconnx import OnnxGraph
from magiconnx.optimize.optimizer_manager import OptimizerManager


def fix_layernorm(graph):
    def merge(sub_node1, sub_node2):
        # sub_node1->next_nodes ==> sub_node2->next_nodes
        for next_node in graph.get_next_nodes(sub_node1.name):
            for idx, input_name in enumerate(next_node.inputs):
                if input_name == sub_node1.outputs[0]:
                    next_node.inputs[idx] = sub_node2.outputs[0]

        # del sub_node1
        graph.del_node(sub_node1.name, auto_connection=False)

    for reducemean_node in graph.get_nodes(op_type="ReduceMean"):
        next_nodes = graph.get_next_nodes(reducemean_node.name)
        if len(next_nodes) == 2 and next_nodes[0].op_type == "Sub" and \
           next_nodes[1].op_type == "Sub":
            sub_node1 = next_nodes[0]
            sub_node2 = next_nodes[1]
            mul_node = graph.get_next_nodes(sub_node1.name)[0]
            mean_node = graph.get_next_nodes(mul_node.name)[0]
            # replace mul with pow
            pow_ini = graph.add_initializer('pow_ini_{}'.format(mul_node.name.split('_')[1]), 
                                            np.array(2).astype('float32'))
            pow_node = graph.add_node('Pow_{}'.format(mul_node.name.split('_')[1]), 'Pow')
            graph.insert_node(sub_node1.name, pow_node, mode='after')
            pow_node.inputs.append(pow_ini.name)
            pow_node.outputs = [mean_node.inputs[0]]
            graph.del_node(mul_node.name, auto_connection=False) 
            # check sub nodes with same inputs and merge sub ops
            if sub_node1.inputs == sub_node2.inputs:
                merge(sub_node1, sub_node2)


def fix_mul(graph):
    # exchange constant value node as second input for mul op
    for mul_node in graph.get_nodes(op_type="Mul"):
        if len(mul_node.inputs) == 2:
            if graph[mul_node.inputs[0]].op_type == "Initializer":
                # exchange input nodes
                input_name1 = mul_node.inputs[0]
                mul_node.inputs[0] = mul_node.inputs[1]
                mul_node.inputs[1] = input_name1


if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    onnx_graph = OnnxGraph(input_path)
    fix_layernorm(onnx_graph)
    fix_mul(onnx_graph)
    optimize_manager_bert = OptimizerManager(onnx_graph, optimizers=["BertBigKernelOptimizer"])
    optimize_manager_bert.apply()
    onnx_graph.save(save_path)
