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

import os
import numpy as np
import PIL.Image as Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _pil_interp
from torchvision import transforms
import argparse
from tqdm import tqdm

def build_transform():
    t = []
    size = 256
    t.append(
        transforms.Resize(size, interpolation=_pil_interp('bicubic')),
    )
    t.append(transforms.CenterCrop(224))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def preprocess(src_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    preprocess = build_transform()

    in_files = sorted(os.listdir(src_path))
    for idx, file in enumerate(tqdm(in_files)):
        input_image = Image.open(src_path + '/' + file).convert('RGB')
        input_tensor = preprocess(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./imageNet/val')
    parser.add_argument('--output_path', type=str, default='./infer/databin/')
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path)

