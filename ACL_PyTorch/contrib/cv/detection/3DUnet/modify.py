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

import numpy as np
from magiconnx import OnnxGraph
import sys

def modify(path, batchsize):
    graph = OnnxGraph(path)
    resizes = graph.get_nodes("Resize")
    #1 128 4 4 4 -> 1 128*4 4 4 -> 1 128*4 8 8 -> 1 128 4 8*8 ->  1 128 8 8*8 -> 1 128 8 8 8       
    shapes1 = [[[1, 128*4, 4, 4], [1,1,2,2], [1, 128, 4, 8*8], [1, 128, 8, 8*8], [1, 128, 8, 8, 8]],
              [[1, 64*8, 8, 8], [1,1,2,2], [1, 64, 8, 16*16], [1, 64, 16, 16*16], [1, 64, 16, 16, 16]],
              [[1, 32*16, 16, 16], [1,1,2,2], [1, 32, 16, 32*32], [1, 32, 32, 32*32], [1, 32, 32, 32, 32]],
              [[1, 16*32, 32, 32], [1,1,2,2], [1, 16, 32, 64*64], [1, 16, 64, 64*64], [1, 16, 64, 64, 64]],
              [[1, 4*16, 16, 16], [1,1,2,2], [1, 4, 16, 32*32], [1, 4, 32, 32*32], [1, 4, 32, 32, 32]],
              [[1, 4*32, 32, 32], [1,1,2,2], [1, 4, 32, 64*64], [1, 4, 64, 64*64], [1, 4, 64, 64, 64]]]

    #4 128 4 4 4       
    shapes4 = [[[4, 128*4, 4, 4], [1,1,2,2], [4, 128, 4, 8*8], [4, 128, 8, 8*8], [4, 128, 8, 8, 8]],
              [[4, 64*8, 8, 8], [1,1,2,2], [4, 64, 8, 16*16], [4, 64, 16, 16*16], [4, 64, 16, 16, 16]],
              [[4, 32*16, 16, 16], [1,1,2,2], [4, 32, 16, 32*32], [4, 32, 32, 32*32], [4, 32, 32, 32, 32]],
              [[4, 16*32, 32, 32], [1,1,2,2], [4, 16, 32, 64*64], [4, 16, 64, 64*64], [4, 16, 64, 64, 64]],
              [[4, 4*16, 16, 16], [1,1,2,2], [4, 4, 16, 32*32], [4, 4, 32, 32*32], [4, 4, 32, 32, 32]],
              [[4, 4*32, 32, 32], [1,1,2,2], [4, 4, 32, 64*64], [4, 4, 64, 64*64], [4, 4, 64, 64, 64]]]
              
    #8 128 4 4 4 -> 8 128*4 4 4 -> 8 128*4 8 8 -> 8 128 4 8*8 ->  8 128 8 8*8 -> 8 128 8 8 8       
    shapes8 = [[[8, 128*4, 4, 4], [1,1,2,2], [8, 128, 4, 8*8], [8, 128, 8, 8*8], [8, 128, 8, 8, 8]],
              [[8, 64*8, 8, 8], [1,1,2,2], [8, 64, 8, 16*16], [8, 64, 16, 16*16], [8, 64, 16, 16, 16]],
              [[8, 32*16, 16, 16], [1,1,2,2], [8, 32, 16, 32*32], [8, 32, 32, 32*32], [8, 32, 32, 32, 32]],
              [[8, 16*32, 32, 32], [1,1,2,2], [8, 16, 32, 64*64], [8, 16, 64, 64*64], [8, 16, 64, 64, 64]],
              [[8, 4*16, 16, 16], [1,1,2,2], [8, 4, 16, 32*32], [8, 4, 32, 32*32], [8, 4, 32, 32, 32]],
              [[8, 4*32, 32, 32], [1,1,2,2], [8, 4, 32, 64*64], [8, 4, 64, 64*64], [8, 4, 64, 64, 64]]]

    #16 128 4 4 4 -> 16 128*4 4 4 -> 16 128*4 8 8 -> 16 128 4 8*8 ->  16 128 8 8*8 -> 16 128 8 8 8 
    shapes16 = [[[16, 128*4, 4, 4], [1,1,2,2], [16, 128, 4, 8*8], [16, 128, 8, 8*8], [16, 128, 8, 8, 8]],
            [[16, 64*8, 8, 8], [1,1,2,2], [16, 64, 8, 16*16], [16, 64, 16, 16*16], [16, 64, 16, 16, 16]],
            [[16, 32*16, 16, 16], [1,1,2,2], [16, 32, 16, 32*32], [16, 32, 32, 32*32], [16, 32, 32, 32, 32]],
            [[16, 16*32, 32, 32], [1,1,2,2], [16, 16, 32, 64*64], [16, 16, 64, 64*64], [16, 16, 64, 64, 64]],
            [[16, 4*16, 16, 16], [1,1,2,2], [16, 4, 16, 32*32], [16, 4, 32, 32*32], [16, 4, 32, 32, 32]],
            [[16, 4*32, 32, 32], [1,1,2,2], [16, 4, 32, 64*64], [16, 4, 64, 64*64], [16, 4, 64, 64, 64]]]

    #32 128 4 4 4 -> 32 128*4 4 4 -> 32 128*4 8 8 -> 32 128 4 8*8 ->  32 128 8 8*8 -> 32 128 8 8 8 
    shapes32 = [[[32, 128*4, 4, 4], [1,1,2,2], [32, 128, 4, 8*8], [32, 128, 8, 8*8], [32, 128, 8, 8, 8]],
            [[32, 64*8, 8, 8], [1,1,2,2], [32, 64, 8, 16*16], [32, 64, 16, 16*16], [32, 64, 16, 16, 16]],
            [[32, 32*16, 16, 16], [1,1,2,2], [32, 32, 16, 32*32], [32, 32, 32, 32*32], [32, 32, 32, 32, 32]],
            [[32, 16*32, 32, 32], [1,1,2,2], [32, 16, 32, 64*64], [32, 16, 64, 64*64], [32, 16, 64, 64, 64]],
            [[32, 4*16, 16, 16], [1,1,2,2], [32, 4, 16, 32*32], [32, 4, 32, 32*32], [32, 4, 32, 32, 32]],
            [[32, 4*32, 32, 32], [1,1,2,2], [32, 4, 32, 64*64], [32, 4, 64, 64*64], [32, 4, 64, 64, 64]]]

    if batchsize == "1":
        shapes = shapes1

    elif batchsize == "4":
        shapes = shapes4
    
    elif batchsize == "8":
        shapes = shapes8
    
    elif batchsize == "16":
        shapes = shapes16
    
    elif batchsize == "32":
        shapes = shapes32
    else:
        print("batchsize输入错误")

    for idx, node in enumerate(resizes):
        print("idx: node.name", idx, node.name)
        reshape1 = graph.add_node(f'Reshape_{node.name}', 'Reshape')
        graph.add_initializer(f'shape_{node.name}', np.array(shapes[idx][0]))
        reshape1.inputs = [node.inputs[0], f'shape_{node.name}']
        reshape1.outputs = [f'Reshape_{node.name}']

        graph[node.inputs[-1]].value = np.array(shapes[idx][1]).astype(np.float32)
        out_name = node.outputs[0]
        node.set_input(0, f'Reshape_{node.name}')
        node.set_output(0, f'{node.name}_reshape')

        reshape2 = graph.add_node(f'Reshape2_{node.name}', 'Reshape')
        graph.add_initializer(f'shape2_{node.name}', np.array(shapes[idx][2]))
        reshape2.inputs = [f'{node.name}_reshape', f'shape2_{node.name}']
        reshape2.outputs = [f'Reshape2_{node.name}_out']

        resize2 = graph.add_node(f'Resize2_{node.name}', 'Resize')
        graph.add_initializer(f'size_{node.name}', np.array(shapes[idx][3]))
        resize2.inputs = [f'Reshape2_{node.name}_out', node.inputs[1], node.inputs[1], f'size_{node.name}']
        resize2.outputs = [f'Resize2_{node.name}']

        reshape3 = graph.add_node(f'Reshape3_{node.name}', 'Reshape')
        graph.add_initializer(f'shape3_{node.name}', np.array(shapes[idx][4]))
        reshape3.inputs = [f'Resize2_{node.name}', f'shape3_{node.name}']
        reshape3.outputs = [out_name]

    graph.save(output_file)

if __name__ == "__main__":
    #input_file是输入的简化后的onnx路径,output_file是输出的onnx名称,batchsize是要转的onnx对应的batchsize大小
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    batch = sys.argv[3]
    modify(input_file, batch)
