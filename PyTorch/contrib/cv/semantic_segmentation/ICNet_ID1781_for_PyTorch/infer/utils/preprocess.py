# Copyright 2022 Huawei Technologies Co., Ltd
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
import argparse
import os
import numpy as np
import PIL.Image as Image

import sys
sys.path.append('..')
from sdk.main import _get_city_pairs
from sdk.main import _mask_transform

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True,
                    help='directory of image to crop')
parser.add_argument('--out_dir', type=str, required=True
                    , help='directory to store the image after being cropped')
args = parser.parse_args()


def crop_imageAndLabel(out_dir, image_path):
    if not os.path.exists(os.path.join(out_dir, "labels")):
        os.makedirs(os.path.join(out_dir, "labels"))

    assert os.path.exists(image_path), "Please put dataset in " + str(image_path)
    images, mask_paths = _get_city_pairs(image_path, 'val')
    assert len(images) == len(mask_paths)
    if not images:
        raise RuntimeError("Found 0 images in subfolders of:" + image_path + "\n")

    for index in range(len(images)):
        mask = Image.open(mask_paths[index])
        mask = _mask_transform(mask)
        mask = mask.astype(np.int32)
        mask = np.expand_dims(mask, 0)#NHW
        filename = images[index].split(os.sep)[-1].split('.')[0]    # get the name of image file
        mask.tofile(os.path.join(os.path.join(out_dir, "labels"), filename+'_label.bin'))

if __name__ == "__main__":
    crop_imageAndLabel(args.out_dir, args.image_path)
