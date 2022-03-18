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
import sys
import shutil
import torch
import numpy as np
from PIL import Image
from glob import glob
import torchvision.transforms as transforms

'''
写在前面的数据预处理文件

数据预处理文件，包括三部分：
1、抽取测试数据集；
2、将图片信息转换成bin文件；
3、生成info文件

注意：此脚本需要放置在数据集同层目录下；
放置结构参考：

'''

dataset_path = sys.argv[1]
bin_path = os.path.join(os.path.abspath('.'),sys.argv[2])
info_path = sys.argv[3]
'''
dataset_pth:数据集路径
bin_path:模型输入的bin路径,路径下是bin文件(在当前目录下）
info_path:模型输入的info路径(不包括文件名）
'''

test_txt_path = os.path.join(dataset_path, 'test.txt')
f = open(test_txt_path,'r')

def pick_test_dataset():
    '''
    从总数据集中抽取对应测试数据集
    '''
    os.makedirs(os.path.join(dataset_path,'Inference_images_dataProcess'))
    images_path = f.read().splitlines()
    images_abs_path = []
    for item in images_path:
        '''
        '/Total_images/'需要修正为真实的完整数据集名称
        '''
        # images_abs_path.append(dataset_path + '/Total_images/' + item)
        images_abs_path.append(dataset_path + '/' + item)
    # print(images_abs_path)
    transfer_path = []
    for item in images_path:
        transfer_path.append(dataset_path + '/Inference_images_dataProcess/' + item.replace('/',''))
    # print(transfer_path)

    image_items = len(images_abs_path)
    for i in range(image_items):
        src = images_abs_path[i]
        dst = transfer_path[i]
        shutil.copyfile(src, dst)
    print('Done')

def jpg_to_bin():
    '''
    将数据缩放到288*800
    '''

    dataset_path = sys.argv[1]
    test_txt_path = os.path.join(dataset_path, 'test.txt')
    f = open(test_txt_path, 'r')
    images_name = f.read().splitlines()

    src_path = []
    for item in images_name:
        src_path.append(dataset_path + '/Inference_images_dataProcess/' + item.replace('/',''))

    bin_floder = os.path.join(bin_path)
    if not os.path.exists(bin_floder):
        os.makedirs(bin_floder)

    dst_path = []
    for item in images_name:
        dst_path.append(bin_floder + '/' + item.replace('/', '').replace('.jpg', '.bin'))
    preprocess = transforms.Compose([
        transforms.Resize((288, 800)),
    ])
    size = len(src_path)
    for i in range(size):
        print(i)
        image = Image.open(src_path[i]).convert('RGB')
        input_tensor = preprocess(image)
        img = np.array(input_tensor).astype(np.uint8)
        img.tofile(dst_path[i])
    print('Done')

def bin_to_info():
    bin_items = glob(os.path.join(sys.argv[2],'*.bin'))
    info_name = os.path.join(info_path, 'test_dataset.info')

    with open(info_name, 'w') as file:
        for index, img in enumerate(bin_items):
            content = ' '.join([str(index), '/'+str(img).replace(dataset_path,''), str(288), str(800)])
            file.write(content)
            file.write('\n')

if __name__=='__main__':
    pick_test_dataset()
    jpg_to_bin()
    bin_to_info()
    print('Preprocess Done.')