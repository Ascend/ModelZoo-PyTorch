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
    parser.add_argument('--model_path', type=str, default='./espnet_trace.ts',
                        help='Original TorchScript model path')
    parser.add_argument('--device_id', type=int, default=0, help='NPU device id')
    # model type: gear, dynamic
    parser.add_argument('--flag', type=str, default='gear', help='Model type')
    return parser.parse_args()


def main():
    args = parse_args()

    torch_aie.set_device(args.device_id)

    model = torch.jit.load(args.model_path)
    model.eval()

    if args.flag == 'gear':
        gear_list = [262, 326, 390, 454, 518, 582, 646, 710, 774, 838, 902, 966, 1028, 1284, 1478]
        inputs = []
        for gear in gear_list:
            inputs.append([torch_aie.Input((gear, 83))])
    elif args.flag == 'dynamic':
        min_shape = (1, 83)
        max_shape = (1500, 83)
        inputs = [torch_aie.Input(min_shape=min_shape, max_shape=max_shape)]
    else:
        raise ValueError('Invalid model type.')

    print('Start compiling model...')
    compiled_model = torch_aie.compile(
        model,
        inputs=inputs,
        precision_policy=_enums.PrecisionPolicy.FP16,
        allow_tensor_replace_int=False,
        soc_version="Ascend310P3")
    print('Model compiled successfully.')
    compiled_model.save(f'./espnet_{args.flag}.ts')

    print('Start exporting om model...')
    torch_aie.export_engine(
        model,
        inputs=inputs,
        precision_policy=_enums.PrecisionPolicy.FP16,
        allow_tensor_replace_int=False,
        soc_version="Ascend310P3",
        method_name="forward",
        path=f'./espnet_{args.flag}.om'
    )
    print('Model exported successfully.')


if __name__ == '__main__':
    main()
