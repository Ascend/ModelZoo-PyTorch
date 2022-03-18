#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#



import random

import cv2
import torchvision.transforms as transforms
# dataloader add 3.0 scale
# dataloader add filer text
from PIL import Image
from torch.utils import data

import util

ctw_root_dir = './data/CTW1500/'
ctw_test_data_dir = ctw_root_dir + 'test/text_image/'

random.seed(123456)


def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img


def scale(img, long_size=1280):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


class CTW1500TestLoader(data.Dataset):
    def __init__(self, long_size=1280):

        data_dirs = [ctw_test_data_dir]

        self.img_paths = []

        for data_dir in data_dirs:
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))
            # img_names.extend(util.io.ls(data_dir, '.gif'))

            img_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

            self.img_paths.extend(img_paths)

        # self.img_paths = self.img_paths[440:]
        # self.gt_paths = self.gt_paths[440:]
        self.long_size = long_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path)

        scaled_img = scale(img, self.long_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)

        return img[:, :, [2, 1, 0]], scaled_img
