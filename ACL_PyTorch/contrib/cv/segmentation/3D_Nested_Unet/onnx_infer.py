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
import time
import numpy as np
import onnxruntime
from tqdm import tqdm


all_time = 0
infer_times = 200
ignore_times = 10
curr_time = 0
assert infer_times > ignore_times

def display_time(func):
    def wrapper(*args):
        t1 = time.time()
        req = func(*args)
        t2 = time.time()
        spent_time = t2 - t1
        print("Single time: {:.4}s".format(spent_time))
        global all_time, curr_time
        curr_time += 1
        if curr_time > ignore_times:
            all_time += spent_time
        return req
    return wrapper


class ONNXModel():
    def __init__(self, onnx_path):
        # providers: TensorrtExecutionProvider/CUDAExecutionProvider/CPUExecutionProvider
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        self.input_feed = None

    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, image_numpy):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_numpy
        self.input_feed = input_feed

    @display_time
    def forward(self):
        self.onnx_session.run(self.output_name, input_feed=self.input_feed)


def create_random_input(input_shape, dtype=np.float32):
    input_data = np.random.random(input_shape).astype(dtype)
    return input_data


if __name__ == '__main__':
    model_file = sys.argv[1]
    input_shape = sys.argv[2]
    np.random.seed(123)
    input_shape = list(map(int, input_shape.split(',')))
    net = ONNXModel(model_file)

    for _ in tqdm(range(infer_times)):
        input_data = create_random_input(input_shape)
        net.get_input_feed(input_data)
        net.forward()

    print("Average time spent: {:.4}s".format(all_time / (infer_times - ignore_times)))
