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
import sys
import argparse
import stat
import pickle as p
import numpy as np
from PIL import Image


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='bytes')
        # take out data in the form of a dict
        X = datadict[b'data']
        Y = datadict[b'fine_labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def create_image_dataset(path):
    """ create Visual dataset """
    imgX, imgY = load_CIFAR_batch(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open('img_label.txt', flags, modes), 'w') as f:
        for i in range(imgY.shape[0]):
            f.write('img' + str(i) + ' ' + str(imgY[i]) + '\n')
    for i in range(imgX.shape[0]):
        imgs = imgX[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB", (i0, i1, i2))
        name = "img" + str(i) + ".png"
        img.save("./pic/" + name, "png")
    print("create visual dataset successfully")


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def preprocess(f_path, b_path):
    r""" trans the image to bin.

    Args:
        f_path : Image file path
        b_path : Output file path

    """

    if not os.path.exists(b_path):
        os.makedirs(b_path)

    imgs_to_label = {}
    with open('./img_label.txt', 'r') as f:
        for line in f.readlines():
            img, label = line.split()[0], line.split()[1]
            imgs_to_label[img] = int(label)

    for img in imgs_to_label.keys():
        file = img + '.png'
        img_raw = Image.open(os.path.join(f_path, file)).convert('RGB')

        # data_ transform
        img = resize(img_raw, 32)
        img = np.array(img, dtype=np.int8)

        img.tofile(os.path.join(b_path, file.split('.')[0] + '.bin'))
    print("create bin dataset successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./cifar-100-python/test')
    parser.add_argument('--bin_path', type=str, default='./bin_data')
    args = parser.parse_args()
    src_path = os.path.realpath(args.src_path)
    bin_path = os.path.realpath(args.bin_path)
    pic_path = "./pic"
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
        create_image_dataset(src_path)
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    preprocess(pic_path, bin_path)