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

center_list = [[140,140],[256,256],[172,114],[128,128],[144,144]]

def performance(args):
    """[Test GPU performance]

    Args:
        args: [parameters]
    """
    time_total = 0
    mean_time_list = []
    img_count = 0
    for i in range(len(center_list)): 
        h, w = center_list[i][0], center_list[i][1]
        f = os.popen('trtexec --onnx={} --fp16 --shapes=lrImage:1x3x{}x{} --workspace=4096'.format(args.onnx_path, h, w))
        gpu_get = f.readlines()[-8:] # 出现GPU Compute是从-8开始 mean time是-5
        for j in range(len(gpu_get)):
            output = gpu_get[j].strip('\n')
            print(output)
        mean = gpu_get[-5] # 出现GPU Compute是从-8开始 mean time是-5
        mean_list = mean.split(' ')
        mean_time = float(mean_list[-2]) # 示例 mean: 3.34409 ms\n 单位是ms
        mean_time_list.append(mean_time)
        time_total = time_total + mean_time # 加和从而后续进行相应的加权平均
        img_count += 1
    time_total = time_total / 1000 # 将ms转换为s
    print("====T4 performance data====")
    print('fps: {}'.format(img_count / time_total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T4 preformance') # T4 preformance parameters
    parser.add_argument('--onnx_path', default='srgan.onnx', type=str, help='onnx path')
    args = parser.parse_args()
    performance(args)
