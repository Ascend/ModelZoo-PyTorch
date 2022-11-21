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
from auto_optimizer import OnnxGraph
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n1',type=str,default='pse.onnx')
parser.add_argument('--n2',type=str,default='pse_new.onnx')
args = parser.parse_args()

g = OnnxGraph.parse(args.n1)
resize_list = g.get_nodes('Resize')
for node in resize_list:
	node['coordinate_transformation_mode'] = 'pytorch_half_pixel'
	node['cubic_coeff_a'] = -0.75
	node['mode'] = 'linear'
	node['nearest_mode'] = 'floor'
	g[node.inputs[1]].value = np.array([], dtype=np.float32)
g.save(args.n2)
