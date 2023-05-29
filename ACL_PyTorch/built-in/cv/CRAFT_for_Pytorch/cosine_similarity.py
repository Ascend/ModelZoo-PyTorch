# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from ais_bench.infer.interface import InferSession
import argparse
import numpy as np
import onnxruntime
import math

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--model_path', default='output/craft_bs1.om', type=str, help='om model path')
parser.add_argument('--bs', default=1, type=int, help='om model bs')
args = parser.parse_args()


def l2norm(inputs):
    '''get L2 norm result'''
    sums = 0
    for data in inputs:
        sums += data * data
    return math.sqrt(sums)

def dot(array1, array2):
    '''get dot result'''
    sums = 0
    idx = 0
    for data1 in array1:
        data2 = array2[idx]
        sums += data1 * data2
        idx += 1
    return sums

def cos_sim(a, b):
    '''get cosine similarity'''
    a_norm = l2norm(a)
    b_norm = l2norm(b)
    cos = dot(a, b) / (a_norm * b_norm)
    return cos


def cal_eval(model_path, bs):
    device = 0
    input_data = np.random.random((bs, 3, 640, 640)).astype("float32")
    net = InferSession(device, model_path)
    output_data = net.infer([input_data])
    y = output_data[0].flatten()
    feature = output_data[1].flatten()
    onnx_model = onnxruntime.InferenceSession("craft.onnx")
    inputs = {
        onnx_model.get_inputs()[0].name: input_data,
    }
    onnx_output = onnx_model.run(None, inputs)
    onnx_y = onnx_output[0].flatten()
    onnx_feature = onnx_output[1].flatten()
    cos_1 = cos_sim(y, onnx_y)
    cos_2 = cos_sim(feature, onnx_feature)
    print("cosine_1:", cos_1)
    print("cosine_2:", cos_2)


if __name__ == '__main__':
    cal_eval(args.model_path, args.bs)

