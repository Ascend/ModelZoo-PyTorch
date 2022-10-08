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
import json
import os
import numpy as np
from PIL import Image
import random

if __name__ == '__main__':
    # 分割后的原始数据集路径
    test_path = "/dataset/valid"
    test_GT_path = "/dataset/valid_GT"
    test_files = os.listdir(test_path)
    test_image_array = []
    test_GT_image_array = []
    for idx, file in enumerate(test_files):
        test_image = Image.open(test_path + '/' + file)
        filename = file.split('_')[-1][:-len(".jpg")]
        test_GT_image = Image.open(test_GT_path + '/' + 'ISIC_' + filename + '_segmentation.png')
        Transform = []
        ResizeRange = random.randint(300, 320)

        test_image = test_image.resize((224, 224), Image.BILINEAR)
        test_image = np.array(test_image)
        test_image = test_image.transpose(2, 0, 1)
        test_image = test_image / 255
        test_image = np.around(test_image, decimals=6)
        test_image = (test_image - 0.5) / 0.5

        p_transform = random.random()

        test_GT_image = test_GT_image.resize((224, 224), Image.BILINEAR)
        test_GT_image = np.array(test_GT_image)
        test_GT_image = test_GT_image / 255
        test_GT_image = np.around(test_GT_image, decimals=6)

        test_image_ = test_image.reshape(-1)
        test_image_array.append(test_image_)
        test_GT_image_ = test_GT_image.reshape(-1)
        test_GT_image_array.append(test_GT_image_)
        print("success ", filename)
    test_image_array = np.vstack(test_image_array)
    np.savetxt('../data/input/valid_images.txt', test_image_array, fmt='%.06f')
    test_GT_image_array = np.vstack(test_GT_image_array)
    np.savetxt('../data/input/valid_GT_images.txt', test_GT_image_array, fmt='%.06f')
