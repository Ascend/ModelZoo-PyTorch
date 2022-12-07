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
import torch
import sys
from PIL import Image
import numpy as np
from torchvision import transforms

def get_img_path(img_folder):
    img_paths = []
    for root, dirs, files in os.walk(img_folder):
        for f in files:
            if f.endswith('.png'):
                print(os.path.join(root, f))
                img_paths.append(os.path.join(root, f))
    return img_paths

def _img_transform(image):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    image = image_transform(image)
    return image


if __name__ == '__main__':

    cityscapes_path = sys.argv[1]
    bin_path = sys.argv[2]
    if os.path.exists(bin_path) is False:
        os.mkdir(bin_path)
        
    split = "val"
    # "./Cityscapes/leftImg8bit/train" or "./Cityscapes/leftImg8bit/val"
    img_folder = os.path.join(cityscapes_path, 'leftImg8bit/' + split)  
    img_paths = get_img_path(img_folder)
    
    for i in range(len(img_paths)):
        filename = os.path.basename(img_paths[i])
        image = Image.open(img_paths[i]).convert('RGB')  # image shape: (W,H,3)
        image = _img_transform(image)  # image shape: (3,H,W) [0,1]
        image = torch.unsqueeze(image, 0)  # image shape: (1,3,H,W) [0,1]        
        #torch.save(image, os.path.join(bin_path, filename.split('.')[0] + '.t')) # save tensor
        image = np.array(image).astype(np.float32)
        image.tofile(os.path.join(bin_path, filename.split('.')[0] + '.bin')) # save bin

