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


parser = argparse.ArgumentParser(description='EDSR preprocess script')
parser.add_argument('-s', required=True, type=str, metavar='PATH',
                    help='path of source image files')
parser.add_argument('-d', required=True, type=str, metavar='PATH',
                    help='path of output')
parser.add_argument('-t', '--data_type', default='float16', type=str,
                    help='data dtype for preprocessed data')
parser.add_argument('--save_img', action='store_true',
                    help='save image')
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
    if not os.path.isdir(os.path.join(save_path, "npy")):
        os.makedirs(os.path.join(save_path, "npy"))
    count = 0
    for image_file in os.listdir(src_path):
        lr_image = imageio.imread(os.path.join(
            src_path, image_file))
        lr_image = np2Tensor(lr_image)
        if args.save_img:
            imageio.imsave(os.path.join(save_path, "png", image_file), np.array(
                lr_image).astype(np.uint8).transpose(1, 2, 0))

        lr_image = np.array(lr_image).astype(np.uint8)
        lr_image = lr_image.astype(args.data_type)

        lr_image.tofile(os.path.join(
            save_path, "bin", image_file.split('.')[0] + ".bin"))
        np.save(
            os.path.join(save_path, "npy", image_file.split('.')[0] + ".npy"),
            lr_image
        )

        count += 1
        print("OK, count = ", count)


def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor


if __name__ == '__main__':
    preprocess(args.s, args.d)
