# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
import math
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str)
parser.add_argument('--label_path', type=str)
parser.add_argument("--target_path", type=str)
parser.add_argument("--start", type=int, default=3002)
parser.add_argument("--end", type=int, default=30002)
opts = parser.parse_args()

target_path = opts.target_path

os.makedirs(target_path, exist_ok=True)

Tags_Attributes = {
    'Bangs': ['with', 'without'],
    'Eyeglasses': ['with', 'without'],
    'HairColor': ['black', 'blond', 'brown'],
}

for tag in Tags_Attributes.keys():
    for attribute in Tags_Attributes[tag]:
        open(os.path.join(target_path, f'{tag}_{attribute}.txt'), 'w')

# celeba-hq
celeba_imgs = opts.img_path
celeba_label = opts.label_path

with open(celeba_label) as f:
    lines = f.readlines()

for line in tqdm.tqdm(lines[opts.start:opts.end]):

    line = line.split()

    filename = os.path.join(os.path.abspath(celeba_imgs), line[0])

    # Use only gender and age as tag-irrelevant conditions. Add other labels if you want.
    if int(line[6]) == 1: 
        with open(os.path.join(target_path, 'Bangs_with.txt'), mode='a') as f:
            f.write(f'{filename} {line[21]} {line[40]}\n')
    elif int(line[6]) == -1:
        with open(os.path.join(target_path, 'Bangs_without.txt'), mode='a') as f:
            f.write(f'{filename} {line[21]} {line[40]}\n')

    if  int(line[16]) == 1:
        with open(os.path.join(target_path, 'Eyeglasses_with.txt'), mode='a') as f:
            f.write(f'{filename} {line[21]} {line[40]}\n')
    elif int(line[16]) == -1:
        with open(os.path.join(target_path, 'Eyeglasses_without.txt'), mode='a') as f:
            f.write(f'{filename} {line[21]} {line[40]}\n')

    if int(line[9]) == 1 and int(line[10]) == -1 and int(line[12]) == -1 and int(line[18]) == -1:
        with open(os.path.join(target_path, 'HairColor_black.txt'), mode='a') as f:
            f.write(f'{filename} {line[21]} {line[40]}\n')
    elif int(line[9]) == -1 and int(line[10]) == 1 and int(line[12]) == -1 and int(line[18]) == -1:
        with open(os.path.join(target_path, 'HairColor_blond.txt'), mode='a') as f:
            f.write(f'{filename} {line[21]} {line[40]}\n')
    elif int(line[9]) == -1 and int(line[10]) == -1 and int(line[12]) == 1 and int(line[18]) == -1:
        with open(os.path.join(target_path, 'HairColor_brown.txt'), mode='a') as f:
            f.write(f'{filename} {line[21]} {line[40]}\n')
    


