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
import cv2


def performance(args):
    """[Test GPU performance]

    Args:
        args: [parameters]
    """
    time_total = 0
    mean_time_list = []
    imgs_list = os.listdir(args.imgs_dir) # images list
    imgs_len = len(imgs_list) # images lenth
    imgs_len_list = [0] * imgs_len # images lenth in each shape
    # if args.height is list, assign args.height to h_list
    h_list, w_list, bs_list = args.height, args.width, args.batch_size
    for i in range(imgs_len):
        img_name = os.path.join(args.imgs_dir, imgs_list[i])
        if img_name.endswith(('jpg', 'png', 'jpeg', 'bmp')):
            print(img_name, "====", i)
            img = cv2.imread(img_name)
            h, w, c = img.shape
            for j in range(len(h_list)):
                if (h_list[j], w_list[j]) == (h, w):
                    imgs_len_list[j] = imgs_len_list[j] + 1
                    break
    imgs_len = sum(imgs_len_list)
    for i in range(len(h_list)): 
        h, w, bs = h_list[i], w_list[i], bs_list[i]
        f = os.popen('trtexec --onnx={}_{}x{}.onnx --fp16 --shapes=image:{}x3x{}x{} --workspace=1024'\
            .format(args.onnx_name, h, w, bs, h, w))
        gpu_get = f.readlines()[-8:] # 出现GPU Compute是从-8开始 mean time是-5
        f.close()
        for j in range(len(gpu_get)):
            output = gpu_get[j].strip('\n')
            print(output)
        mean = gpu_get[-5] # 出现GPU Compute是从-8开始 mean time是-5
        mean_list = mean.split(' ')
        mean_time = float(mean_list[-2]) # 示例 mean: 3.34409 ms\n 单位是ms
        mean_time_list.append(mean_time)
        time_total = time_total + mean_time * imgs_len_list[i] # 加和从而后续进行相应的加权平均
    for i in range(len(mean_time_list)):
        print('number:{} mean time:{} ms'.format(imgs_len_list[i], mean_time_list[i]))
    time_total = time_total / 1000 # 将ms转换为s
    print("====T4 performance data====")
    print('fps: {}'.format(sum(imgs_len_list) / (time_total / bs))) # 最后还要除以batch_size才是真实的fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T4 preformance') # T4 preformance parameters
    parser.add_argument('--onnx_name', default='rcf_change_sim', type=str, help='onnx name')
    parser.add_argument('--imgs_dir', default='data/BSR/BSDS500/data/images/test', type=str, 
    help='images path')
    parser.add_argument('--batch_size', nargs='+',
                        type=int, help='batch size')
    parser.add_argument('--height', nargs='+',
                        type=int, help='input height')
    parser.add_argument('--width', nargs='+',
                        type=int, help='input width')
    args = parser.parse_args()
    performance(args)
