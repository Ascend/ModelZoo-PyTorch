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

import sys
import os
import numpy as np
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _pil_interp
from PIL import Image

sys.path.append('./Swin-Transformer')
from main import parse_option


def build_transform(config):
    resize_im = config.DATA.IMG_SIZE > 32
    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def preprocess(config):
    Transform = build_transform(config)
    val_path = os.path.join(config.DATA.DATA_PATH, 'val')
    save_path = config.BIN_PATH
    val_files = os.listdir(val_path)
    i = 0
    for val_set in val_files:
        valset_p = os.path.join(val_path, val_set)
        if not os.path.isdir(valset_p):
            i = i + 1
            file = val_set
            print(file, "===", i)
            input_image = Image.open(valset_p).convert('RGB')
            input_tensor = Transform(input_image)
            img = np.array(input_tensor).astype(np.float32)
            img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))
            continue
        files = os.listdir(valset_p)
        for file in files:
            i = i + 1
            print(file, "===", i)
            input_image = Image.open(valset_p + '/' + file).convert('RGB')
            input_tensor = Transform(input_image)
            img = np.array(input_tensor).astype(np.float32)
            img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


if __name__ == '__main__':
    _, config = parse_option()
    preprocess(config)
