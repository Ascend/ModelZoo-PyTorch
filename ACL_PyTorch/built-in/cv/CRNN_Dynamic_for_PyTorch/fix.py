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
from auto_optimizer.graph_refactor.interface.base_node import BaseNode, Node, Initializer


def fix_gemm(graph):
    # reshape+gemm+reshape -> matmul+add
    for gemm_node in graph.get_nodes(op_type="Gemm"):
       
        # prev
        reshape_0_node = graph.get_prev_node(gemm_node.inputs[0])
        concat_0_node = graph.get_prev_node(reshape_0_node.inputs[1])

        concat_f_node = graph.get_prev_node(reshape_0_node.inputs[0])
        
        graph.remove(reshape_0_node.name)
        graph.remove(concat_0_node.name)


        unsqu_0_node = graph.get_prev_node(concat_0_node.inputs[0])
        unsqu_1_node = graph.get_prev_node(concat_0_node.inputs[1])

        graph.remove(unsqu_0_node.name)
        graph.remove(unsqu_1_node.name)

        # Unsqueeze0
        mul_0_node = graph.get_prev_node(unsqu_0_node.inputs[0])
        gather_0_node = graph.get_prev_node(mul_0_node.inputs[0])
        shape_0_node = graph.get_prev_node(gather_0_node.inputs[0])

        graph.remove(mul_0_node.name)
        graph.remove(gather_0_node.name)        
        graph.remove(shape_0_node.name)

        # Unsqueeze1
        gather_2_node = graph.get_prev_node(unsqu_1_node.inputs[0])
        shape_1_node = graph.get_prev_node(gather_2_node.inputs[0])

        graph.remove(gather_2_node.name)
        graph.remove(shape_1_node.name)

        # next 
        reshape_1_node = graph.get_next_nodes(gemm_node.outputs[0])[0]


        concat_1_node = graph.get_prev_node(reshape_1_node.inputs[1])

        graph.remove(reshape_1_node.name)
        graph.remove(concat_1_node.name)


        unsqu_2_node = graph.get_prev_node(concat_1_node.inputs[0])
        unsqu_3_node = graph.get_prev_node(concat_1_node.inputs[1])

        graph.remove(unsqu_2_node.name)
        graph.remove(unsqu_3_node.name)

        # Unsqueeze3
        gather_1_node = graph.get_prev_node(unsqu_3_node.inputs[0])
        shape_2_node = graph.get_prev_node(gather_1_node.inputs[0])

        graph.remove(gather_1_node.name)
        graph.remove(shape_2_node.name)
        
        # add
        add_init = gemm_node.inputs[2]

        add_node = graph.add_node(f"Add_after_{concat_f_node.name}",op_type="Add")
        
        graph.insert_node(concat_f_node.name, add_node)
        add_node.inputs.append(add_init)

        # matmul
        matmul_node = graph.add_node(
                gemm_node.name.replace("Gemm", "MatMul"),
                "MatMul",
            )
        
        graph.insert_node(f"Add_after_{concat_f_node.name}", matmul_node, mode='before')

        matmul_init_value = np.array(graph[gemm_node.inputs[1]].value.T,dtype=np.float32)



        matmul_init = graph.add_initializer(
            f"{matmul_node.name}_value",
            matmul_init_value
        )
        matmul_node.inputs.append(matmul_init.name)

        graph.remove(gemm_node.name)
        
        
def fix_lstm(graph):
    
    # 删除 init_c 和 init_h


    for ind, constant in enumerate(graph.get_nodes('ConstantOfShape')):

        
            
        concat_node = graph.get_prev_node(constant.inputs[0])
        unsqueeze_node = graph.get_prev_node(concat_node.inputs[1])
        gather_node = graph.get_prev_node(unsqueeze_node.inputs[0])
        shape_node = graph.get_prev_node(gather_node.inputs[0])

        graph.remove(constant.name,{})
        graph.remove(concat_node.name)
        graph.remove(unsqueeze_node.name)
        
        if gather_node.op_type == "Gather":
            graph.remove(gather_node.name)
        if shape_node.op_type == "Shape":
            graph.remove(shape_node.name)
            
    
    
    # 删除slice
    for lstm in graph.get_nodes("LSTM"):

        # 删除后两个输入
        lstm.inputs = lstm.inputs[:-2]
        


        

        slice_prev_node = graph.get_prev_node(lstm.inputs[0])
        squeeze_node = graph.get_next_nodes(lstm.outputs[0])[0]

        if slice_prev_node.op_type == "Slice":
            slice_next_node = graph.get_next_nodes(squeeze_node.outputs[0])[0]

            graph.remove(slice_prev_node.name)
            graph.remove(slice_next_node.name)

            lstm.attrs['direction'] = 'reverse'

        graph.remove(squeeze_node.name)




    # 合并squeeze
    
    for ind, concat in enumerate(graph.get_nodes("Concat")):

       
        concat.attrs['axis'] = 3

        squeeze_node = graph.add_node(f"Squeeze_after_{concat.name}",op_type="Squeeze",attrs={'axes':[1]})
        graph.insert_node(concat.name, squeeze_node)
    

if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]

    onnx_graph = OnnxGraph.parse(input_path)
    fix_gemm(onnx_graph)
    fix_lstm(onnx_graph)
    onnx_graph.save(save_path)