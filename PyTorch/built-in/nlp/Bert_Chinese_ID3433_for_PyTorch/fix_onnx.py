# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import sys
from copy import deepcopy
import numpy as np
from magiconnx import OnnxGraph
from magiconnx.optimize.optimizer_manager import OptimizerManager


FP16_MAX_VALUE = 65504
FP16_MIN_VALUE = -65504


def fix_mul(graph):
    for mul_node in graph.get_nodes(op_type='Mul'):
        for input_name in mul_node.inputs:
            input_node = graph[input_name]
            if input_node.op_type == "Constant" or input_node.op_type == "Initializer":
                fixed_value = deepcopy(input_node.value)
                value_mask_pos = (fixed_value > FP16_MAX_VALUE)
                value_mask_neg = (fixed_value < FP16_MIN_VALUE)
                if np.sum(value_mask_pos) > 0 or np.sum(value_mask_neg) > 0:
                    print(f"Fix value node: {input_name}")
                    fixed_value[value_mask_pos] = FP16_MAX_VALUE
                    fixed_value[value_mask_neg] = FP16_MIN_VALUE
                    input_node.value = fixed_value


if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    onnx_graph = OnnxGraph(input_path)
    fix_mul(onnx_graph)
    optimize_manager_bert = OptimizerManager(onnx_graph, optimizers=["BertBigKernelOptimizer"])
    optimize_manager_bert.apply()
    onnx_graph.save(save_path)
