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

import pickle
import argparse
'''read pkl'''
# path = 'conformer-base-result.pkl'
# f = open(path, 'rb')
# data = pickle.load(f)
# # print(data)
# # print(len(data))
# # print(type(data))
# print(data['class_scores'])
# print(data['pred_label'])
# print(data['pred_class'])
# for i in data:
#     print(i)

import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
def preprocess(src_path, save_path):
  preprocess = transforms.Compose([
      transforms.Resize(256, interpolation=Image.BICUBIC),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  in_files = os.listdir(src_path)
  for i in tqdm(range(len(in_files))):
        for idx, file in enumerate(in_files):
            idx = idx + 1
            #print(file, "===", idx)
            input_image = Image.open(src_path + '/' + file).convert('RGB')
            input_tensor = preprocess(input_image)
            img = np.array(input_tensor).astype(np.float32)
            img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))

def parse_args():
    parser = argparse.ArgumentParser(description='datasets preprocess')
    parser.add_argument('--src_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    preprocess(src_path=args.src_path,
                save_path=args.save_path)

