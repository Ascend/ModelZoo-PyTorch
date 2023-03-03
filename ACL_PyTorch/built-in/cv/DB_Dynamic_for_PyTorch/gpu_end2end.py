
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

import logging

import os
import sys
import argparse
import time
from tqdm import tqdm


import numpy as np

sys.path.append('./Tensorrt_dynamic_api/')
from tensorrt_dynamic.builder import Builder
from tensorrt_dynamic.infer import Infer




logger = logging.getLogger(__name__)


def build_session(onnx_path, engine_path):

    precision = "fp16"  # fp16 or int8

    input_shapes = {
        "input": {
            "min_shapes": [1, 3, 640, 640],
            "opt_shapes": [1, 3, 1280, 1280],
            "max_shapes": [1, 3, 4096, 4096]
        },
    }

    builder = Builder(onnx_path, engine_path, input_shapes, precision)
    builder.set_calibration_dataset(None)
    status = builder.build()
    if status:
        print("model build success .")
    else:
        print("model build failed .")
        return None

    max_shape = [[1, 3, 4096, 4096]]

    infer = Infer(engine_path, max_shape)
    return infer


def main(data_path, session):

    imgs = os.listdir(data_path)
    datas = [[np.load(os.path.join(data_path,img))] for img in imgs]

    t = time.time()
    for data in tqdm(datas):
        result = session.infer(data)
    print(f'GPU compute time {time.time() - t} s')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='infer E2E')
    parser.add_argument("--data_path", default="./pre_npy", help='data path')
    parser.add_argument("--onnx_path", default="./dbnet_fix.onnx", help='onnx path')
    parser.add_argument("--engine_path", default="./dbnet.trt", help='engine path')
    flags = parser.parse_args()
    
    global_session = build_session(flags.onnx_path, flags.engine_path)
    
    main(flags.data_path, global_session)
