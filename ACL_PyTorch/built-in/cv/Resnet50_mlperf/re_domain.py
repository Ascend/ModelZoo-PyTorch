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

import os
import sys
from auto_optimizer import OnnxGraph

if __name__ == '__main__':
  try:
    # original onnx path
    input_onnx = sys.argv[1] 

    # destany path
    output_onnx = sys.argv[2]
  except IndexError:
    print("Stopped!")
    exit(1)
  if not (os.path.exists(input_onnx)):
    print("original onnx path does not exist.")
  
  print(input_onnx)
  
  graph = OnnxGraph.parse(input_onnx)
  graph.save(output_onnx)
