# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse

from tqdm import tqdm
import numpy as np
import PIL.Image as pil_image
from torch.serialization import save
import torchvision.transforms as transforms


def preprocess(src_path, save_path):
    # create dir
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(os.path.join(save_path, 'png')):
        os.makedirs(os.path.join(save_path, 'png'))
    if not os.path.isdir(os.path.join(save_path, 'bin')):
        os.makedirs(os.path.join(save_path, 'bin'))

    for image_file in tqdm(os.listdir(src_path)):
        src_img = os.path.join(src_path, image_file)
        image = pil_image.open(src_img).convert('RGB')
        image = transforms.Pad(padding=(0, 0, 256 - image.size[0], 
            256 - image.size[1]), padding_mode='edge')(image)
        image.save(os.path.join(save_path, 'png', 
            image_file).replace('.png', '_256.png'))
        image = np.array(image).astype(np.float32).transpose() / 255
        image.tofile(os.path.join(save_path, 'bin', 
            image_file.split('.')[0] + '.bin'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description='SRFlow preprocess script')
    parser.add_argument('-s', '--source', type=str, metavar='PATH',
                        help='path to source image files.')
    parser.add_argument('-o', '--output', type=str, metavar='PATH',
                        help='path to save output binary files.')
    args = parser.parse_args()
    preprocess(args.source, args.output)
