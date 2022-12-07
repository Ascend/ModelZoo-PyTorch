# Copyright 2020 Huawei Technologies Co., Ltd
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
from PIL import Image
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize
)
import sys
import os
from tqdm import tqdm


def build_transforms(
    height,
    width,
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    **kwargs
):
    """Builds test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    normalize = Normalize(mean=norm_mean, std=norm_std)
    

    transform_test = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])
    return transform_test


# dataset processing
def preprocess(src_path, save_path):
    i = 0
    in_files = os.listdir(src_path)
    for file in tqdm(in_files):
        file_d = os.path.join(src_path, file)
        tp = file_d.split('.')[-1]
        if tp == 'jpg':
            i = i + 1
            input_image = Image.open(src_path + '/' + file).convert('RGB')
            height = 256
            width = 128
            transform_test = build_transforms(height, width, 
                         norm_mean = [0.485, 0.456, 0.406], norm_std = [0.229, 0.224, 0.225])
            output_image = transform_test(input_image)
            output_tensor = np.array(output_image).astype(np.float32)
            output_tensor.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


if __name__ == '__main__': 
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    preprocess(src_path, save_path)