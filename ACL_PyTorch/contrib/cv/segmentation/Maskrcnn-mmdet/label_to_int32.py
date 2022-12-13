# Copyright 2020 Huawei Technologies Co., Ltd
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
import onnx


def getNodeByName(nodes, name: str):
    for n in nodes:
        if n.name == name:
            return n
    
    return -1


if __name__ == '__main__':
    if len(sys.argv) is not 3:
        print('only need 2 params, include onnx source path and dest path.')

    model = onnx.load(sys.argv[1])
    cast = getNodeByName(model.graph.node, 'Cast_622')
    cast.attribute[0].i = 6
    onnx.save(model, sys.argv[2])
    print("onnx saved to: {}".format(sys.argv[2]))
