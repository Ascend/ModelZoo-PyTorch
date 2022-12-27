# Copyright 2022 Huawei Technologies Co., Ltd
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
import cv2
import numpy as np

sys.path.append('LPRNet_Pytorch')

from tqdm import tqdm
from LPRNet_Pytorch.data.load_data import CHARS, CHARS_DICT

IMG_SIZE = [94, 24]  # input shape: w:94, h:24
LPR_MAX_LEN = 8  # License Plate max length


class LabelError(Exception):
    def __init__(self, image_name: str) -> None:
        super(LabelError, self).__init__()
        self.image_name = image_name

    def __str__(self) -> str:
        return self.image_name


def save_bin(bin: np.ndarray, bin_name: str, dst_path: str = './prep_data') -> None:
    """save numpy binary data to file"""
    if not bin_name.endswith('.bin'):
        bin_name += '.bin'

    bin_path = os.path.join(dst_path, bin_name)
    bin.tofile(bin_path)


def transform(image: np.ndarray) -> np.ndarray:
    """image normalization processing and transposing"""
    image = image.astype('float32')
    image -= 127.5
    image *= 0.0078125  # / 128
    image = np.transpose(image, (2, 0, 1))  # 24, 94, 3 --> 3, 24, 94
    return image


def label_check(label):
    if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
            and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
        print("Error label, please check!")
        return False
    else:
        return True


def gen_image_bin(image_path: str):
    """convert jpg to bin, return fp32 image data and label"""
    # read and transform image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    if height != IMG_SIZE[1] or width != IMG_SIZE[0]:
        image = cv2.resize(image, IMG_SIZE)
    image = transform(image)

    # get and check label
    image_name, suffix = os.path.splitext(os.path.basename(image_path))
    image_name = image_name.split('-')[0].split('_')[0]
    label = []
    for c in image_name:
        label.append(CHARS_DICT[c])
    if len(label) == 8 and label_check(label) == False:
        print(f'image: [{image_name}] label error!')
        raise LabelError(image_name)

    return image, label


def preprocess(src_path: str, dst_path: str) -> None:
    """read images from src_path, convert jpg to bin and save in dst_path"""
    images_list = os.listdir(src_path)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    print('Start preprocessing images...')
    for image_name in tqdm(images_list):
        image_path = os.path.join(src_path, image_name)
        try:
            image, label = gen_image_bin(image_path)
        except LabelError:
            continue

        save_bin(image, image_name.split('.')[0], dst_path)

    print(f'Images process done, see bins in {dst_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', dest='img_path', default='./LPRNet_Pytorch/data/test',
                        help='license plate images path')
    parser.add_argument('--dst_path', dest='dst_path', default='./prep_data',
                        help='processed image binary data store path')
    args = parser.parse_args()
    preprocess(args.img_path, args.dst_path)
