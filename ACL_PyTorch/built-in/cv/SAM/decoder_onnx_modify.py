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


"""
修改原因：/output_upscaling/output_upscaling.1/Div 节点对应的LayerNorm算子存在精度问题，
修改方式：通过atc关闭融合的方式失败，所以通过此脚本插入一个不影响计算结果的Pow(1)算子，
         破坏图结构使atc转换时匹配不到LayerNorm的融合规则，以此破坏融合。
"""

import sys
import numpy as np
from auto_optimizer import OnnxGraph


if __name__ == '__main__':
    input_onnx = sys.argv[1]
    output_onnx = sys.argv[2]

    g = OnnxGraph.parse(input_onnx)
    g.add_initializer('exponent', np.array(1.0, dtype=np.float32))
    new_pow = g.add_node('NewPow', 'Pow', inputs=['', ''], outputs=[])
    g.insert_node('/output_upscaling/output_upscaling.1/Div', new_pow, refer_index=0, mode='before')
    new_pow.inputs.append('exponent')
    g.update_map()
    g.save(output_onnx)