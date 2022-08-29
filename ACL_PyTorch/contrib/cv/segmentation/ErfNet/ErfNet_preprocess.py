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

import sys
import os
import shutil
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm


def getFileList(dir, Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


def main():
    if os.path.isdir(output_dir1):
        shutil.rmtree(output_dir1)
        os.makedirs(output_dir1)
    if not os.path.isdir(output_dir1):
        os.makedirs(output_dir1)

    img_list = getFileList(input_dir1, [], 'png')
    val_transform = Compose([
        Resize(512, Image.BILINEAR),
        ToTensor(),
    ])
    for i in tqdm(range(len(img_list))):
        if len(img_list[i]) == 0: continue
        img_name = os.path.splitext(os.path.basename(img_list[i]))[0]
        input_img = Image.open(img_list[i]).convert('RGB')
        img_tensor = val_transform(input_img)
        img = np.array(img_tensor, dtype=np.float32)
        img.tofile(os.path.join(output_dir1, img_name + ".bin"))
    print("测试集处理完成")

    source_path = os.path.abspath(input_dir2)
    target_path = os.path.abspath(output_dir2)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if os.path.exists(source_path):
        for root, dirs, files in os.walk(source_path):
            for i in tqdm(range(len(files))):
                src_file = os.path.join(root, files[i])
                shutil.copy(src_file, target_path)

    print('验证集处理完成')


if __name__ == "__main__":
    input_dir1 = sys.argv[1]  # /opt/npu/cityscapes/leftImg8bit/val
    output_dir1 = sys.argv[2]  # ./prep_dataset
    input_dir2 = sys.argv[3]  # /opt/npu/cityscapes/gtFine/val
    output_dir2 = sys.argv[4]  # ./gt_label/
    main()
