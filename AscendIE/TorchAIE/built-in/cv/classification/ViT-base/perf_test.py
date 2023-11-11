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
from torch_aie import _enums


def parse_args():
    parser = argparse.ArgumentParser(description='SwinTransformer Evaluation.')
    parser.add_argument('--model_path', type=str, default='./vit_base_patch8_224.ts',
                        help='Original TorchScript model path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--optim_level', type=int, default=0, help='Optimization level')
    return parser.parse_args()


def main():
    args = parse_args()

    torch_aie.set_device(0)

    model = torch.jit.load(args.model_path)
    model.eval()

    input_info = [torch_aie.Input((args.batch_size, 3, args.image_size, args.image_size))]
    print('Start compiling model.')
    print(f'Using optimization level {args.optim_level}')
    compiled_model = torch_aie.compile(
        model,
        inputs=input_info,
        precision_policy=_enums.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        optimization_level=args.optim_level,
        soc_version="Ascend310P3")
    print('Model compiled successfully.')

    random_input = torch.rand(args.batch_size, 3, args.image_size, args.image_size)

    stream = torch_aie.npu.Stream("npu:0")
    random_input = random_input.to("npu:0")

    # warm up
    num_warmup = 10
    for _ in range(num_warmup):
        with torch_aie.npu.stream(stream):
            compiled_model(random_input)
            stream.synchronize()

    # inference
    start = time.time()
    print('inference...')
    num_infer = 100
    for _ in range(num_infer):
        with torch_aie.npu.stream(stream):
            compiled_model(random_input)
            stream.synchronize()
    avg_time = (time.time() - start) / num_infer
    fps = args.batch_size / avg_time
    print(f'FPS: {fps:.2f}')


if __name__ == '__main__':
    main()
