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
import os
import torch
import sys
sys.path.append('./PraNet')
import numpy as np
import torchvision.transforms as transforms
from utils.dataloader import test_dataset
from utils.dataloader import get_loader


def main(image_root, gt_root, testsize, save_path):
    test_loader = test_dataset(image_root, gt_root, testsize)

    for i in range(test_loader.size):
        i=0
        image, gt, name = test_loader.load_data()
        image = np.array(image).astype(np.float32)
        if not(os.path.exists(save_path)):
            os.makedirs(save_path, exist_ok=True)
        image.tofile(os.path.join(save_path, name.split('.')[0] + ".bin"))

if __name__ == "__main__":
    data_path = sys.argv[1]
    images_path = '{}/images/'.format(data_path)
    gts_path = '{}/masks/'.format(data_path)
    testsize = 352
    save_path = sys.argv[2]
    main(images_path, gts_path, testsize, save_path)
