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
import numpy as np
import PIL.Image as pil_image
from torch.serialization import save
import torchvision.transforms as transforms
import os
import argparse

parser = argparse.ArgumentParser(description='SRFlow preprocess script')
parser.add_argument('-s', default='', type=str, metavar='PATH',
                    help='path of source image files (default: none)')
parser.add_argument('-d', default='', type=str, metavar='PATH',
                    help='path of output (default: none)')
args = parser.parse_args()


def preprocess(src_path, save_path):
    # create dir
    if not os.path.isdir(src_path):
        os.makedirs(src_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(os.path.join(save_path, "png")):
        os.makedirs(os.path.join(save_path, "png"))
    if not os.path.isdir(os.path.join(save_path, "bin")):
        os.makedirs(os.path.join(save_path, "bin"))
    for image_file in os.listdir(src_path):
        if not "_256.png" in image_file:
            image = pil_image.open(os.path.join(
                src_path, image_file)).convert('RGB')
            print('size', image.size)
            image = transforms.Pad(padding=(
                0, 0, 256-image.size[0], 256-image.size[1]), padding_mode='edge')(image)
            image.save(os.path.join(
                save_path, "png", image_file).replace('.png', '_256.png'))

            image = np.array(image).astype(np.float32).transpose()/255

            image.tofile(os.path.join(
                save_path, "bin", image_file.split('.')[0] + ".bin"))
            print('shape', image.shape)
            print("OK")


if __name__ == '__main__':
    preprocess(args.s, args.d)
