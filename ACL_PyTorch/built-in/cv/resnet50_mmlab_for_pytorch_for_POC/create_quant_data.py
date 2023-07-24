# Copyright 2023 Huawei Technologies Co., Ltd
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
import argparse
import stat
import pickle as p
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


model_config = {
    'resize': 32,
    'mean': [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
    'std': [0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
}


def resize(img, size, interpolation=Image.BILINEAR):
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


def amct_input_bin(pic_path):
    in_files = sorted(os.listdir(src_path))
    image_name = in_files[0]
    file_path = os.path.join(src_path, image_name)
    if os.path.isdir(file_path):
        image_name = os.listdir(file_path)[0]
        file_path = os.path.join(file_path, image_name)
    imgs_to_label = {}
    with open('./img_label.txt', 'r') as f:
        for line in f.readlines():
            img, label = line.split()[0], line.split()[1]
            imgs_to_label[img] = int(label)

    for img_name in imgs_to_label.keys():
        file = img_name + '.png'
        img = cv2.imread(os.path.join(pic_path, file))
        img = resize(img, model_config['resize']) # Resize
        img = img.transpose(2, 0, 1) # ToTensor: HWC -> CHW
        img = np.expand_dims(img, 0)
        img = img/255. # ToTensor: div 255
        img = np.vstack([img]*24).astype(dtype=np.float32)
        img -= np.array(model_config['mean'], dtype=np.float32)[:, None, None] # Normalize: mean
        img /= np.array(model_config['std'], dtype=np.float32)[:, None, None] # Normalize: std
        img.tofile(os.path.join(save_path, image_name.split('.')[0] + ".bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./cifar-100-python/test')
    parser.add_argument('--save_path', type=str, default='./amct_bin_data')
    parser.add_argument('--amct', action='store_true')
    args = parser.parse_args()
    src_path = os.path.realpath(args.src_path)
    save_path = os.path.realpath(args.save_path)
    pic_path = "./pic"
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
        create_image_dataset(src_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.amct:
        amct_input_bin(pic_path)


if __name__ == '__main__':
    main()
    