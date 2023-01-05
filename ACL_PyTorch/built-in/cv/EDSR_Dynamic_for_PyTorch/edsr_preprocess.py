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
import imageio
import argparse
import torch
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(description='EDSR preprocess script')
parser.add_argument('-s', required=True, type=str, metavar='PATH',
                    help='path of source image files')
parser.add_argument('-d', required=True, type=str, metavar='PATH',
                    help='path of output')
parser.add_argument('-t', '--data_type', default='float16', type=str,
                    help='data dtype for preprocessed data')
args = parser.parse_args()


def preprocess(src_path, save_path):
    for image_file in tqdm(os.listdir(src_path)):
        lr_image = imageio.imread(os.path.join(
            src_path, image_file))
        lr_image = np2Tensor(lr_image)
        lr_image = np.array(lr_image).astype(np.uint8)
        lr_image = lr_image.astype(args.data_type)
        np.save(
            os.path.join(save_path, image_file.split('.')[0] + ".npy"),
            np.expand_dims(lr_image, axis=0)
        )


def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor


if __name__ == '__main__':
    preprocess(args.s, args.d)
