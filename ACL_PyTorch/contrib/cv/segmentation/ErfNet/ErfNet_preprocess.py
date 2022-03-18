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

def getFileList(dir,Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
    
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, Filelist, ext)
 
    return Filelist

def main():

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
 
    img_list = getFileList(input_dir, [], 'png')    

    val_transform = Compose([
        Resize(512, Image.BILINEAR),
        ToTensor(),
    ])

    for img_id in img_list:
        if len(img_id) == 0: continue
        img_name = os.path.splitext(os.path.basename(img_id))[0]
        input_img = Image.open(img_id).convert('RGB')
        img_tensor = val_transform(input_img)
        img = np.array(img_tensor).astype(np.float32)
        img.tofile(os.path.join(output_dir, img_name + ".bin"))

if __name__ == "__main__":
    input_dir = sys.argv[1] # '/home/common/jyf/cityscapes/leftImg8bit/val'
    output_dir = sys.argv[2] # './prep_dataset''
    main()
