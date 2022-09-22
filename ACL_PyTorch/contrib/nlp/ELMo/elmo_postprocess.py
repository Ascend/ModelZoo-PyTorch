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
import onnx
import onnxruntime as ort
import pdb
import argparse



def cosine_similarity(x, y):
    x1 = x.flatten().astype(np.float64)
    y1 = y.flatten().astype(np.float64)
    dot = np.dot(x1, y1)
    lx = np.linalg.norm(x1)
    ly = np.linalg.norm(y1)
    cos = dot / (lx * ly)
    return cos


parser = argparse.ArgumentParser()
parser.add_argument('--onnx_model', default='elmo_sim.onnx')
parser.add_argument('--onnx_input', default='bin_path/')
parser.add_argument('--om_output', default='om_out/2022_09_17-01_26_57/')
opt = parser.parse_args()

onnx_input_path = opt.onnx_input
om_output_path = opt.om_output
onnx = onnx.load_model(opt.onnx_model)
sess = ort.InferenceSession(onnx.SerializeToString())
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

similarity = 0
for i in range(15947):
    onnx_input_file = np.fromfile(onnx_input_path + '{0}.bin'.format(i), dtype='int32').reshape((1, 8, 50))
    om_output_file = np.fromfile(om_output_path + '{0}_0.bin'.format(i), dtype='float32').reshape((1, 8, 1024))
    onnx_output = sess.run([output_name], {input_name : onnx_input_file})
    cosine_sim = cosine_similarity(om_output_file, onnx_output[0])
    print(i, "  cosine_similarity: ", cosine_sim)
    similarity += cosine_sim
print('average similarity: ', similarity / 15947)
    
