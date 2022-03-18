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

import cv2
from PIL import Image
import os
import transformer as T
import torch
import argparse

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
parser.add_argument('--datasets', default=r'/opt/npu/coco/val2017', type=str)
parser.add_argument('--img_file', default=r'img_file', type=str)
parser.add_argument('--mask_file', default=r'mask_file', type=str)
args = parser.parse_args()
print(args)

if not os.path.exists(args.img_file):
    os.mkdir(args.img_file)
if not os.path.exists(args.mask_file):
    os.mkdir(args.mask_file)

normalize = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

detr_transformer = T.Compose([
    T.RandomResize([768], max_size=1400),
    normalize,
])
coco_val = args.datasets
files = os.listdir(coco_val)
files.sort()
shape = [[768, 1280], [768, 768], [768, 1024], [1024, 768], [1280, 768], [768, 1344], [1344, 768], [1344, 512],
         [512, 1344]]
mask = [[24, 40], [24, 24], [24, 32], [32, 24], [40, 24], [24, 42], [42, 24], [32, 16], [16, 32]]
for i in shape:
    bin_file = '{}/{}_{}'.format(args.img_file, i[0], i[1])
    mask_file = '{}/{}_{}_mask'.format(args.mask_file, i[0], i[1])
    if not os.path.exists(bin_file):
        os.mkdir(bin_file)
    if not os.path.exists(mask_file):
        os.mkdir(mask_file)

cou = 0
for file in files:
    cou += 1
    img_path = os.path.join(coco_val, file)
    img = cv2.imread(img_path)
    h, w = img.shape[0], img.shape[1]
    input_size = (h, w)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(img)
    img_tensor = detr_transformer(pilimg)
    mask_data = torch.zeros([1, int(img_tensor.shape[1] / 32), int(img_tensor.shape[2] / 32)], dtype=torch.bool)
    img_save_path = r'{}/{}_{}'.format(args.img_file, img_tensor.shape[1], img_tensor.shape[2])
    img_tensor.numpy().tofile(os.path.join(img_save_path, file.replace('.jpg', '.bin')))
    mask_save_path = r'{}/{}_{}_mask'.format(args.mask_file, img_tensor.shape[1], img_tensor.shape[2])
    mask_data.numpy().tofile(os.path.join(mask_save_path, file.replace('.jpg', '.bin')))
    print(cou)