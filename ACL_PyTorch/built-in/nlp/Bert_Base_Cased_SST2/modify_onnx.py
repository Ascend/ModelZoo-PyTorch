# Copyright 2023 Huawei Technologies Co., Ltd
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
import sys

from magiconnx import OnnxGraph
from magiconnx.optimize.optimizer_manager import OptimizerManager


if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    onnx_graph = OnnxGraph(input_path)
    optimizer = OptimizerManager(
        onnx_graph,
        optimizers=["BertBigKernelOptimizer"]
    )
    optimizer.apply()
    onnx_graph.save(save_path)
