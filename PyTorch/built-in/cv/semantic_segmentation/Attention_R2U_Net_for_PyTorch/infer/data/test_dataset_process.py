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
import json
import os
import numpy as np
from PIL import Image
from torchvision import transforms as T
import random

if __name__ == '__main__':
    test_path = "dataset/test"
    test_GT_path = "dataset/test_GT"
    test_files = os.listdir(test_path)
    test_image_array = []
    test_GT_image_array = []
    for idx, file in enumerate(test_files):
        test_image = Image.open(test_path + '/' + file)
        filename = file.split('_')[-1][:-len(".jpg")]
        test_GT_image = Image.open(test_GT_path + '/' + 'ISIC_' + filename + '_segmentation.png')
        Transform = []
        ResizeRange = random.randint(300,320)
        Transform.append(T.Resize((224,224)))
        p_transform = random.random()
        Transform.append(T.Resize((224,224)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        test_image = Transform(test_image)
        test_GT_image = Transform(test_GT_image)
        print(test_GT_image.shape)
        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        test_image = Norm_(test_image)
        test_image_ = test_image.reshape(-1)
        test_image_array.append(test_image_)
        test_GT_image_ = test_GT_image.reshape(-1)
        test_GT_image_array.append(test_GT_image_)
        print("success ",filename)
    test_image_array = np.vstack(test_image_array)
    np.savetxt('infer/data/input/test_images.txt', test_image_array, fmt='%.06f')
    test_GT_image_array = np.vstack(test_GT_image_array)
    np.savetxt('infer/data/input/test_GT_images.txt', test_GT_image_array, fmt='%.06f')
