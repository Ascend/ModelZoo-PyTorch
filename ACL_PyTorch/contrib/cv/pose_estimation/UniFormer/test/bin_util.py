# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import numpy as np
from argparse import ArgumentParser


def pack(input_path, output_path, meta_path, batch_size):
    os.makedirs(output_path, exist_ok=True)
    file_list = os.listdir(input_path)
    meta = []

    for i in range(0, len(file_list), batch_size):
        batch_list = file_list[i:i+batch_size]
        batch_list = batch_list + (batch_size - len(batch_list)) * [batch_list[-1]]
        batch_img = []
        for file in batch_list:
            img = np.fromfile(os.path.join(input_path, file), dtype=np.float32)
            batch_img.append(img)
        
        np.array(batch_img, dtype=np.float32).tofile(os.path.join(output_path, f'{i//batch_size}.bin'))
        meta.append(','.join(batch_list))
    
    with open(meta_path, 'w') as f:
        f.writelines([f'{m}\n' for m in meta])


def unpack(input_path, output_path, meta_path, batch_size):
    latest_result = os.listdir(input_path)
    latest_result.sort()
    input_path = os.path.join(input_path, latest_result[-1])
    output_path = os.path.join(output_path, latest_result[-1])
    os.makedirs(output_path, exist_ok=True)

    lines = []
    with open(meta_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.rstrip()
        batch_img = np.fromfile(os.path.join(input_path, f'{i}_output_0.bin'), dtype=np.float32)
        batch_img = np.reshape(batch_img, [batch_size, -1])
        for img, name in zip(batch_img, line.split(',')):
            img.tofile(os.path.join(output_path, f'{name[:-4]}_output_0.bin'))


def main():
    parser = ArgumentParser()
    parser.add_argument('--pack', help='pack bin files to batch',
                        action='store_true')
    parser.add_argument('--unpack', help='unpack batch to bin files',
                        action='store_true')
    parser.add_argument('--input', help='input bin file path')
    parser.add_argument('--output', help='output bin file path')
    parser.add_argument('--meta', help='meta file path',
                        default='meta.txt')
    parser.add_argument('--batch_size', help='batch size',
                        type=int, default=1)
    args = parser.parse_args()

    if args.pack:
        pack(args.input, args.output, args.meta, args.batch_size)
    elif args.unpack:
        unpack(args.input, args.output, args.meta, args.batch_size)


if __name__ == '__main__':
    main()
