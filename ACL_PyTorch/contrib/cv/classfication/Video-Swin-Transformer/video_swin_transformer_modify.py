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
import sys
from magiconnx import OnnxGraph
from magiconnx.optimize.optimizer_manager import OptimizerManager

def improve_model(onnx_path, save_path):
    graph = OnnxGraph(onnx_path)
    graph.del_node('Slice_15866',auto_connection=False)
    graph.del_node('Slice_15869', auto_connection=False)
    graph.del_node('Concat_15870', auto_connection=False)
    graph.del_node('Concat_15886', auto_connection=False)
    graph.del_node('Slice_15882', auto_connection=False)
    graph.del_node('Slice_15885', auto_connection=False)
    graph['Reshape_15920'].node.input[0] = graph['Concat_15854'].node.output[0]
    graph['Add_16084'].node.input[1] = graph['Concat_16051'].node.output[0]
    graph.del_node('Slice_16063', auto_connection=False) 
    graph.del_node('Slice_16066',auto_connection=False)
    graph.del_node('Concat_16067', auto_connection=False)
    graph.del_node('Slice_16079', auto_connection=False)
    graph.del_node('Slice_16082', auto_connection=False)
    graph.del_node('Concat_16083', auto_connection=False)
    optimize_manager_cus2 = OptimizerManager(graph, optimizers=['Int64ToInt32Optimizer'])
    optimized_graph = optimize_manager_cus2.apply()
    optimized_graph.save(save_path)
    
if __name__ == '__main__':
    if len(sys.argv) is not 3:
        print('only need 2 params, include onnx source path and dest path.')
    else:
        improve_model(sys.argv[1], sys.argv[2])