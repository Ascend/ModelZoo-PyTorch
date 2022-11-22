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

import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm


def save_image(image_numpy_path, image_path):
    image_pil = Image.fromarray(image_numpy_path)
    image_pil.save(image_path)

if __name__ == "__main__":
    bin_dir = sys.argv[1]
    save_image_dir = sys.argv[2]
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    bin_path_list = os.listdir(bin_dir)

    for iterm in tqdm(bin_path_list):
        bin_path = os.path.join(bin_dir, iterm)
        image = np.fromfile(bin_path, dtype = np.float32)
        image.shape = 3, 1024, 2048  
        image = image.transpose(1, 2, 0) 
        image = (image + 1.0) / 2.0 * 255.0
        image_numpy = np.clip(image, 0, 255)
        image_numpy = image_numpy.astype(np.uint8)
        save_path = os.path.join(save_image_dir, iterm.split('.')[0] + 'generated.jpg')
        save_image(image_numpy, save_path)
