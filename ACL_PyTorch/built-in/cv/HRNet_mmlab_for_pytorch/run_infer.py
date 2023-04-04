# Copyright 2023 Huawei Technologies Co., Ltd
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
import os


def parse_args():
    parser = argparse.ArgumentParser('hrnet infer', add_help=False)
    parser.add_argument('--data_path', default='data_file')
    parser.add_argument('--out_put', default='out_put')
    parser.add_argument('--result', default='result')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--device_id', default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    arg = parse_args()
    res1 = os.path.join(arg.result, 'res1')
    res2 = os.path.join(arg.result, 'res2')
    if not os.path.exists(arg.out_put):
        os.mkdir(arg.out_put)

    if not os.path.exists(arg.result):
        os.mkdir(arg.result)
        os.mkdir(res1)
        os.mkdir(res2)

    data_path1 = os.path.join(arg.data_path, 'data1')
    data_path2 = os.path.join(arg.data_path, 'data2')
    all_files = os.listdir(data_path1)
    shape_list = []
    for file in all_files:
        shape = file.split('_')
        shape_list.append([shape[0], shape[1]])

    command = 'python3.7 -m ais_bench --model "model/hrnet_bs{}.om" --input "{}/{}_{}" --output "{}" --outfmt NPY ' \
              '--dymDims x:{},3,{},{} --device {}'
    for i in shape_list:
        command1 = command.format(arg.batch_size, data_path1, i[0], i[1], arg.out_put, arg.batch_size, i[0], i[1],
                                  arg.device_id)
        os.system(command1)
    mv_command = 'mv {}/*/* {}'.format(arg.out_put, res1)
    os.system(mv_command)

    if not os.path.exists(arg.out_put):
        os.mkdir(arg.out_put)

    for i in shape_list:
        command2 = command.format(arg.batch_size, data_path2, i[0], i[1], arg.out_put, arg.batch_size, i[0], i[1],
                                  arg.device_id)
        os.system(command2)
    mv_command = 'mv {}/*/* {}'.format(arg.out_put, res2)
    os.system(mv_command)


if __name__ == '__main__':
    main()
