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

import os
import json
import argparse

def performance_calculation(save_path):
    save_path_file = os.listdir(save_path)
    length = 0
    sum_length = 0
    for file_name in save_path_file:
        if not file_name.endswith(".json"):
            continue
        with open(os.path.join(save_path, file_name), mode="r") as f:
            result = json.load(f)
            sum_length = sum_length + result["throughput"]
            length += 1
    return sum_length / length


def statistics(args):
    src_path_dir = os.listdir(args.src_path)
    for dir_name in src_path_dir:
        dir_name = os.path.join(args.src_path, dir_name)
        W, H = (dir_name.split("_")[-2:])
        os.system(r"{} --model={} --input={} --output={} --dymHW {},{} --batchsize={}  --device={} ".
            format(args.interpreter, args.om_path, dir_name, args.save_path, H, W, args.batchsize, args.device))
    throughput = performance_calculation(args.save_path)
    print("-----------------------------------------------------------")
    print(f"the throughput data of {args.om_path} is FPS:{throughput}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='performance statistics script')
    parser.add_argument('--interpreter', default='python3 -m ais_bench', type=str,
                        help='path of interpreter')
    parser.add_argument('--om_path', default='./srgan_bs1.om', type=str,
                        help='path of source om model ')
    parser.add_argument('--src_path', default='./preprocess_data', type=str,
                        help='path of source image files')
    parser.add_argument('--save_path', default='./result/bs1', type=str,
                        help='path of output')
    parser.add_argument('--batchsize', default='1', type=str,
                        help='batchsize of om model')
    parser.add_argument('--device', default='1', type=str,
                        help='npu device ID')
    args_paeser = parser.parse_args()
    
    if not os.path.exists(args_paeser.save_path):
        os.makedirs(args_paeser.save_path)
    else:
        os.system(f"rm {args_paeser.save_path}/* -rf")

    statistics(args_paeser)