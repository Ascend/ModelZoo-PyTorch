# Copyright 2020 Huawei Technologies Co., Ltd
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
import sys
from PIL import Image
import numpy as np
import multiprocessing

model_config = {
    'resnet': {
        'resize': 256,
        'centercrop': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'inceptionv3': {
        'resize': 342,
        'centercrop': 299,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'inceptionv4': {
        'resize': 342,
        'centercrop': 299,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
    },
}


def center_crop(img, output_size):
    if isinstance(output_size, int):  # 可以判断一个变量的类型
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    # 一般情况也是使用四舍五入的规则，但是碰到.5的这样情况，如果要取舍的位数前的小数是奇数，则直接舍弃，如果偶数这向上取舍。
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
    # 你输入的大小部分截取出来，其他的没有任何变化


def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)  # resize是原位操作，直接将原来的a3拉成一维，


def gen_input_bin(mode_type, file_batches, batch):
    i = 0
    for file in file_batches[batch]:
        i = i + 1
        print("batch", batch, file, "===", i)

        # RGBA to RGB
        image = Image.open(os.path.join(src_path, file)).convert('RGB')
        image = resize(image, model_config[mode_type]['resize'])  # Resize
        image = center_crop(image, model_config[mode_type]['centercrop'])  # CenterCrop
        img = np.array(image, dtype=np.float32)
        img = img.transpose(2, 0, 1)  # ToTensor: HWC -> CHW 矩阵转置
        img = img / 255.  # ToTensor: div 255
        img -= np.array(model_config[mode_type]['mean'], dtype=np.float32)[:, None, None]  # Normalize: mean
        img /= np.array(model_config[mode_type]['std'], dtype=np.float32)[:, None, None]  # Normalize: std
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin")) # tofile()将数组中的数据以二进制格式写进文件


def preprocess(mode_type, src_path, save_path): # 数据预处理
    files = os.listdir(src_path)
    file_batches = [files[i:i + 500] for i in range(0, 50000, 500) if files[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))  # 进程是文件批的长度
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(mode_type, file_batches, batch))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise Exception("usage: python3 xxx.py [model_type] [src_path] [save_path]")
    mode_type = sys.argv[1]
    src_path = sys.argv[2]
    save_path = sys.argv[3]
    src_path = os.path.realpath(src_path) # 获取当前执行脚本的绝对路径。
    save_path = os.path.realpath(save_path)
    if mode_type not in model_config:
        model_type_help = "model type: "
        for key in model_config.keys():
            model_type_help += key
            model_type_help += ' '
        raise Exception(model_type_help)
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    preprocess(mode_type, src_path, save_path)
