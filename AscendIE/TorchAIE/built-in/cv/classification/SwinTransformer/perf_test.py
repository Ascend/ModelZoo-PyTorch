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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./swin_base_patch4_window12_384_aie.ts',
                        help='Compiled model path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--image_size', type=int, default=384, help='Image size')
    parser.add_argument('--device_id', type=int, default=0, help='NPU device id')
    return parser.parse_args()


def main():
    args = parse_args()

    torch_aie.set_device(args.device_id)

    model = torch.jit.load(args.model_path)
    model.eval()
    print('Model loaded successfully.')

    random_input = torch.rand(args.batch_size, 3, args.image_size, args.image_size)
    device = f'npu:{args.device_id}'
    random_input = random_input.to(device)
    stream = torch_aie.npu.Stream(device)

    # warm up
    num_warmup = 10
    for _ in range(num_warmup):
        with torch_aie.npu.stream(stream):
            model(random_input)
            stream.synchronize()
    print('warmup done')

    # performance test
    print('Start performance test.')
    num_infer = 100
    start = time.time()
    for _ in range(num_infer):
        with torch_aie.npu.stream(stream):
            model(random_input)
            stream.synchronize()
    avg_time = (time.time() - start) / num_infer
    fps = args.batch_size / avg_time
    print(f'FPS: {fps:.4f}')


if __name__ == '__main__':
    main()
