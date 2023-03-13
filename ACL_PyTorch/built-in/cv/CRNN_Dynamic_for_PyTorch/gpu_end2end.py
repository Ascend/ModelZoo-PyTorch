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

import argparse
import time
import numpy as np
from Tensorrt_dynamic_api.builder import Builder
from Tensorrt_dynamic_api.infer import Infer

total_img = 3000

def build_session(onnx_path, engine_path):
    precision = 'fp16'

    input_shapes = {
        "imgs":{
            "min_shapes": [1,1,32,32],
            "opt_shapes": [4,1,32,100],
            "max_shapes": [16,1, 32, 6400]
        },
    }

    builder = Builder(onnx_path, engine_path, input_shapes, precision)
    builder.set_calibration_dataset(None)
    status = builder.build()
    if status:
        print('model build success.')
    else:
        print("model build failed.")
        return None
    
    max_shape = [[16,1,32,6400]]
    session_ = Infer(engine_path, max_shape)
    return session_

def get_gpu_time(input_npy, session):
    tt = 0
    for index in range(total_img):
        index += 1
        data = np.load(f"{input_npy}/test_{index}.npy")
        t0 = time.time()
        result = session.infer([data])
        tt += (time.time() - t0)
    
    '''time(s)'''
    print("*"*50)
    print(f"GPU E2E time(s): {tt}s")
    print("*"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npy', default="./input_npy", type=str,
                        help="input path")
    parser.add_argument('--onnx_path', default="./crnn.onnx",
                        type=str, help='onnx path')
    parser.add_argument('--engine_path', default='crnn.trt',
                        type=str, help='trt save path')
    args = parser.parse_args()

    global_session = build_session(args.onnx_path, args.engine_path)
    get_gpu_time(args.input_npy, global_session)