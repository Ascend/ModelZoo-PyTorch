# Copyright 2022 Huawei Technologies Co., Ltd
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
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable

def preprocess_file(input_path: str, output_path: str):
    print("input_path", input_path)
    img = Image.open(input_path).convert('RGB')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transformer = transforms.Compose([
        transforms.Scale(513),
        transforms.CenterCrop(513),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = val_transformer(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0).float()
    img_tensor = Variable(img_tensor, requires_grad=False)
    img_tensor.reshape(1, 3, 513, 513)
    img_numpy = img_tensor.cpu().numpy()

    img_name = input_path.split('/')[-1]
    bin_name = img_name.split('.')[0] + ".bin"
    output_fl = os.path.join(output_path, bin_name)   
    # save img_tensor as binary file for om inference input
    img_numpy.tofile(output_fl)

if __name__ == "__main__":
    input_img_dir = sys.argv[1]
    output_img_dir = sys.argv[2]
    input_list = sys.argv[3]
    with open(os.path.join(input_list), "r") as f:
        file_names = [x.strip() for x in f.readlines()]
    
    images = [os.path.join(input_img_dir, x + ".jpg") for x in file_names]
    for image_name in images:
        if not image_name.endswith(".jpg"):
            continue
        print("start to process image {}....".format(image_name))
        preprocess_file(image_name, output_img_dir)