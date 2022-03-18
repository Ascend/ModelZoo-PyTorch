# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import shutil
from libtiff import TIFF    # pip install libtiff
from scipy import misc
import random


def tif2png(_src_path, _dst_path):
    """
    Usage:
        formatting `tif/tiff` files to `jpg/png` files
    :param _src_path:
    :param _dst_path:
    :return:
    """
    tif = TIFF.open(_src_path, mode='r')
    image = tif.read_image()
    misc.imsave(_dst_path, image)


def data_split(src_list):
    """
    Usage:
        randomly spliting dataset
    :param src_list:
    :return:
    """
    counter_list = random.sample(range(0, len(src_list)), 550)

    return counter_list


if __name__ == '__main__':
    src_dir = '../Dataset/train_dataset/CVC-EndoSceneStill/CVC-612/test_split/masks_tif'
    dst_dir = '../Dataset/train_dataset/CVC-EndoSceneStill/CVC-612/test_split/masks'

    os.makedirs(dst_dir, exist_ok=True)
    for img_name in os.listdir(src_dir):
        tif2png(os.path.join(src_dir, img_name),
                os.path.join(dst_dir, img_name.replace('.tif', '.png')))
