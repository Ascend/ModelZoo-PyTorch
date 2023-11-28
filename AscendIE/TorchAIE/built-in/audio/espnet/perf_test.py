# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

import argparse
import time

import torch
import torch_aie
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./espnet_gear.ts',
                        help='Compiled model path')
    parser.add_argument('--device_id', type=int, default=0, help='NPU device id')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    torch_aie.set_device(args.device_id)
    device = f'npu:{args.device_id}'
    stream = torch_aie.npu.Stream(device)

    model = torch.jit.load(args.model_path)
    model.eval()
    print('Model loaded successfully.')

    # warm up
    num_warmup = 20
    random_input = torch.rand(1478, 83).to(device)
    for _ in range(num_warmup):
        with torch_aie.npu.stream(stream):
            model(random_input)
            stream.synchronize()
    print('warmup done')

    # performance test
    print('Start performance test.')
    num_infer_per_shape = 20
    shapes = [262, 326, 390, 454, 518, 582, 646, 710, 774, 838, 902, 966, 1028, 1284, 1478]
    shape_num = [96, 682, 1260, 1230, 1052, 940, 656, 462, 303, 207, 132, 67, 38, 48, 3]
    shape_t = []
    total_time = 0
    FPS = 0
    for shape in shapes:
        cur_time = 0
        random_input = torch.rand(shape, 83).to(device)
        for i in range(num_infer_per_shape):
            with torch_aie.npu.stream(stream):
                infer_start = time.time()
                model(random_input)
                stream.synchronize()
                infer_end = time.time()
                cur_time += infer_end - infer_start
        shape_t.append(cur_time / num_infer_per_shape)
    total_time = np.multiply(np.array(shape_t), np.array(shape_num))
    total_time = total_time.tolist()
    fps = 1 / (sum(total_time) / 7176)
    print("fps:", fps)
