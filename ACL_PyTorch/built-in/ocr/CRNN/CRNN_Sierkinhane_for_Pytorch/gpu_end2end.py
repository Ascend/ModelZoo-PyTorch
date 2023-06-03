
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8

import os
import time
import argparse
from tqdm import tqdm

import numpy as np

from tensorrt_dynamic.infer import Infer
from tensorrt_dynamic.builder import Builder


def gen_data(data_path):
    
    imgs = os.listdir(data_path)

    for img in imgs:
        
        yield np.load(os.path.join(data_path,img)), img

def build_session(onnx_path, engine_path):
    precision = "fp16"  # fp16 or int8

    input_shapes = [
        {
            "min_shapes": [1, 1, 32, 32],
            "opt_shapes": [8, 1, 32, 576],
            "max_shapes": [64, 1, 32, 2048]
        },
    ]

    builder = Builder(onnx_path, engine_path, input_shapes, precision)
    builder.set_calibration_dataset(None)
    status = builder.build()
    if status:
        print("model build success .")
    else:
        print("model build failed .")
    

    session = Infer(engine_path)
    return session

def get_gpu_time(data_path, session, output):
   
    tt = 0
    num_data = 0
    for data, name in tqdm(gen_data(data_path)):
    
        t0 = time.time()
        result = session.infer([data])
        tt += (time.time() - t0)
        save_path = os.path.join(output,f'{name[:-4]}_0.npy')
        np.save(save_path, result['output'])
        num_data += 1

        
    print('*'*50)
    print(f"GPU E2E time(s): {tt} s")
    print(f"GPU fps: {1000 / tt} fps")
    print('*'*50)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='infer E2E')
    parser.add_argument("--data_path", default="./images/preprocessed_dym_test_cj", help='data path')
    parser.add_argument("--trt_engine", default="./crnn.trt", help='trt path')
    parser.add_argument("--onnx_path", default="./crnn.onnx", help='onnx path')
    parser.add_argument("--output", default="./gpu_result", help='result save path')
    
    flags = parser.parse_args()
    if not os.path.exists(flags.output):
        os.mkdir(flags.output)

    trt_session = build_session(flags.onnx_path, flags.trt_engine)
    get_gpu_time(flags.data_path, trt_session, flags.output)
