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

#-*- coding:utf-8 -*-
import os
import argparse
import config


def performance(args):
    """[Test GPU performance]

    Args:
        args: [parameters]
    """
    time_total = 0
    mean_time_list = []
    for i in range(config.center_len):
        h, w = config.center_list[i][0], config.center_list[i][1]
        f = os.popen('trtexec --onnx={}_{}x{}.onnx --fp16 --shapes=image:1x3x{}x{} --workspace=1024'.format(args.onnx_path, h, w, h, w))
        gpu_get = f.readlines()[-8:] # 获取后8行的输出。
        for j in range(len(gpu_get)):
            output = gpu_get[j].strip('\n')
            print(output)
        mean = gpu_get[-8] # 出现GPU Compute mean time是第-8行
        mean_list = mean.split(',')[-3].split(' ')  # 获取mean: *** 字符串
        mean_time = float(mean_list[-2])  # 获取mean后面的数字, 示例 mean: 3.34409  单位是ms
        mean_time_list.append(mean_time)
        time_total = time_total + mean_time * config.center_count[i] # 加和从而后续进行相应的加权平均
    for i in range(len(mean_time_list)):
        print('number:{} mean time:{} ms'.format(config.center_count[i], mean_time_list[i]))
    time_total = time_total / 1000 # 将ms转换为s
    print("====T4 performance data====")
    print('fps: {}'.format(config.imgs_len / time_total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T4 preformance') # T4 preformance parameters
    parser.add_argument('--onnx_path', default='ctpn_change.onnx', type=str, help='onnx path')
    args = parser.parse_args()
    performance(args)
