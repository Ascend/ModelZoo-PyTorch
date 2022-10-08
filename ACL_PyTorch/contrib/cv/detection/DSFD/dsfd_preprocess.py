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

#coding=utf-8

import os
from PIL import Image
import numpy as np
from glob import glob
from torchvision import datasets, transforms
import argparse

parser = argparse.ArgumentParser(description="trans pth to onnx usage")
parser.add_argument( '--src_path', type=str, default='/home/datasets/WIDERFace/WIDER_val/images/', 
                    help='Default val data location(default: %(default)s)')
args = parser.parse_args()

def img2bin(src_path, save_path):
  preprocess = transforms.Compose([
      transforms.Resize(256, Image.BICUBIC),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  i = 0
  in_files = os.listdir(src_path)
  for file in in_files:
      i = i + 1
      print(file, "===", i)
      files = os.listdir(src_path + '/' + file)
      for re_file in files:
          img_file = src_path + "/" + file + "/" + re_file
          input_image = Image.open(img_file).convert('RGB')
          input_tensor = preprocess(input_image)
          img = np.array(input_tensor).astype(np.float32)
          img.tofile(os.path.join(save_path, re_file.split('.')[0] + ".bin"))

def bin2info(bin_dir, info_data, width, height):
    bin_images = glob(os.path.join(bin_dir, '*.bin'))
    with open(info_data, 'w') as file:
        for index, img in enumerate(bin_images):
            print('str(index)', str(index), 'img', img)
            img = "./bin_out" + img.split("bin_out")[1]
            content = ' '.join([str(index), img, str(width), str(height)])
            file.write(content)
            file.write('\n')

if __name__ == "__main__":
    
    bin_path = "./bin_out/"
    info_path = "info_result.info"
    img2bin(args.src_path, bin_path)
    bin2info(bin_path, info_path, 224, 224)