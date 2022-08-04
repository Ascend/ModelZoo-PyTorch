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
import glob
import shutil

root_dir = os.environ['DATASETS']

# 首先如果没有根目录创建根目录
if not os.path.exists(os.path.join(root_dir, "cityscapes")):
    os.mkdir(os.path.join(root_dir, "cityscapes"))

# 之后创建下面的三个子目录
temp_dir = os.path.join(root_dir, "cityscapes")

train_img_dir = os.path.join(temp_dir, "train_img")
train_inst_dir = os.path.join(temp_dir, "train_inst")
train_label_dir = os.path.join(temp_dir, "train_label")


if not os.path.exists(train_img_dir):
    os.mkdir(train_img_dir)
if not os.path.exists(train_inst_dir):
    os.mkdir(train_inst_dir)
if not os.path.exists(train_label_dir):
    os.mkdir(train_label_dir)

# 下面完成了train_img目录的复制粘贴
original_img_dir = os.path.join(root_dir, "leftImg8bit")
train_original_img_dir = os.path.join(original_img_dir, "train")

city_name_dir = os.listdir(train_original_img_dir)
img_number = 0
for city_name in city_name_dir:
    temp_city_dir = os.path.join(train_original_img_dir, city_name)
    trian_img_list = glob.glob(os.path.join(temp_city_dir, "*.png"))
    img_number += len(trian_img_list)
    for img in trian_img_list:
        shutil.copy(img, train_img_dir)
print("train_img 共有{}张！".format(img_number))  # 2975张

# 下面对另外的两个目录进行赋值粘贴
original_gtfine_dir = os.path.join(root_dir, "gtFine")
train_original_gtfine_dir = os.path.join(original_gtfine_dir, "train")

city_name_dir = os.listdir(train_original_gtfine_dir)
img_inst_number = 0
img_label_number = 0

for city_name in city_name_dir:
    temp_city_dir = os.path.join(train_original_gtfine_dir, city_name)
    trian_gtfine_list = glob.glob(os.path.join(temp_city_dir, "*.png"))
    # print(trian_gtfine_list)

    for img in trian_gtfine_list:
        # print(img[-9:])
        if img[-9:] == "ceIds.png":
            img_inst_number += 1
            shutil.copy(img, train_inst_dir)
        elif img[-9:] == "elIds.png":
            img_label_number += 1
            shutil.copy(img, train_label_dir)

    print("inst_img 共有{}张！".format(img_inst_number))
    print("label_img 共有{}张！".format(img_label_number))
print("inst和label图片全部复制完毕!")
print("inst_img 共有{}张！".format(img_inst_number))
print("label_img 共有{}张！".format(img_label_number))