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
from copy import deepcopy
import onnx
from onnx import (helper, TensorProto)
from onnx.onnx_ml_pb2 import ModelProto
from magiconnx import OnnxGraph
import numpy as np


batch_size = sys.argv[1]

graph = OnnxGraph(f'onnx_sim_dir/textcnn_{batch_size}bs_sim.onnx')




graph.del_node('Squeeze_5',{0:0},auto_connection=True)
graph.del_node('Squeeze_11',{0:0},auto_connection=True)
graph.del_node('Squeeze_17',{0:0},auto_connection=True)

Maxpool_1 = graph.add_node('maxpool_1',
                  'MaxPool',
                  {'ceil_mode': 0, 'kernel_shape': [31,1], 'pads': 0, 'strides':[31,1]})
graph['MaxPool_6'] = Maxpool_1


Maxpool_2 = graph.add_node('maxpool_2',
                  'MaxPool',
                  {'ceil_mode': 0, 'kernel_shape': [30,1], 'pads': 0, 'strides':[30,1]})
graph['MaxPool_12'] = Maxpool_2


Maxpool_3 = graph.add_node('maxpool_3',
                  'MaxPool',
                  {'ceil_mode': 0, 'kernel_shape': [29,1], 'pads': 0, 'strides':[29,1]})
graph['MaxPool_18'] = Maxpool_3

graph.del_node('Squeeze_7',{0:0},auto_connection=True)
graph.del_node('Squeeze_13',{0:1},auto_connection=True)
graph.del_node('Squeeze_19',{0:2},auto_connection=True)




squeeze = graph.add_node('squeeze_1',
                     'Squeeze',
                   {'axis': [2,3]})
graph.insert_node('Gemm_21', squeeze, mode='before')


graph.save(f'mg_onnx_dir/textcnn_{batch_size}bs_mg.onnx')
