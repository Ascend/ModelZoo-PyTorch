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
import cv2
from PIL import Image
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable


def mobilenet_onnx(input_path: str, output_path: str):
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transformer = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224)
    ])

    img_tensor = val_transformer(pilimg)
    img_tensor = np.array(img_tensor)
    img_tensor = torch.from_numpy(img_tensor)

    img_tensor = torch.unsqueeze(img_tensor, dim=0).float()
    img_tensor = Variable(img_tensor, requires_grad=False)

    img_numpy = img_tensor.cpu().numpy()
    img_numpy = img_numpy.astype(np.int8)
    img_name = input_path.split('/')[-1]
    bin_name = img_name.split('.')[0] + ".bin"
    output_fl = os.path.join(output_path, bin_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # save img_tensor as binary file for om inference input
    img_numpy.tofile(output_fl)


if __name__ == "__main__":
    input_img_dir = sys.argv[1]
    output_img_dir = sys.argv[2]
    images = os.listdir(input_img_dir)
    for image_name in images:
        temp = os.listdir(input_img_dir + image_name)
        for image in temp:
            image_end = image.split('.')[1].lower()
            if not image_end.endswith("jpeg"):
                continue
            print("start to process image {}....".format(image))
            path_image = os.path.join(input_img_dir+image_name, image)
            mobilenet_onnx(path_image, output_img_dir)
