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
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import sys

def get_input_bin(src_path, save_path,width, height):
  width,height = int(width), int(height)
  preprocess = transforms.Compose([
      transforms.Resize([width,height]),
      transforms.ToTensor(),
  ])
  in_files = os.listdir(src_path)
  for idx, file in enumerate(in_files):
      idx = idx + 1
      print(file, "===", idx)
      input_image = Image.open(src_path + '/' + file)
      input_image=input_image.convert('RGB')
      input_tensor = preprocess(input_image)
      img = np.array(input_tensor).astype(np.float32)
      #print(img.shape,img.dtype)
      img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))

if __name__ == '__main__':
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    width = sys.argv[3]
    height = sys.argv[4]
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    get_input_bin(src_path, save_path, width, height)
      
