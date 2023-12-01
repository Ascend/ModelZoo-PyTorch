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

import torch
import torch_aie
from torch_aie import _enums


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./swin_base_patch4_window12_384.ts',
                        help='Original TorchScript model path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--image_size', type=int, default=384, help='Image size')
    parser.add_argument('--output_path', type=str, default='./swin_base_patch4_window12_384_aie.ts',
                        help='Compiled model path')
    parser.add_argument('--optim_level', type=int, default=0, help='Optimization level')
    parser.add_argument('--device_id', type=int, default=0, help='NPU device id')
    return parser.parse_args()


def main():
    args = parse_args()

    torch_aie.set_device(args.device_id)

    model = torch.jit.load(args.model_path)
    model.eval()

    input_info = [torch_aie.Input((args.batch_size, 3, args.image_size, args.image_size))]
    print('Start compiling model.')
    print(f'Optimization level: {args.optim_level}')
    compiled_model = torch_aie.compile(
        model,
        inputs=input_info,
        precision_policy=_enums.PrecisionPolicy.FP16,
        optimization_level=args.optim_level,
        soc_version="Ascend310P3")
    print('Model compiled successfully.')
    compiled_model.save(args.output_path)


if __name__ == '__main__':
    main()
