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
import sys
import cv2
from glob import glob


def get_bin_info(file_path, info_name, width, height):
    bin_images = glob(os.path.join(file_path, '*.bin'))
    print(bin_images)
    with open(info_name, 'w') as file:
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, width, height])
            file.write(content)
            file.write('\n')


def get_png_info(image_names, info_name):
    with open(info_name, 'w') as file:
        for image_name in image_names:
            print(image_name)
            if len(image_names) == 0:
                continue
            else:
                for index, png in enumerate(image_name):
                    img_cv = cv2.imread(png)
                    shape = img_cv.shape
                    width, height = shape[1], shape[0]
                    content = ' '.join([str(index), png, str(width), str(height)])
                    file.write(content)
                    file.write('\n')

def get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        # "./Cityscapes/leftImg8bit/train" or "./Cityscapes/leftImg8bit/val"
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        # "./Cityscapes/gtFine/train" or "./Cityscapes/gtFine/val"
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        # img_paths与mask_paths的顺序是一一对应的
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths

    return img_paths, mask_paths

if __name__ == '__main__':
    file_type = sys.argv[1]
    file_path = sys.argv[2]
    info_name = sys.argv[3]
    res = get_city_pairs(file_path)
    if file_type == 'bin':
        width = sys.argv[4]
        height = sys.argv[5]
        assert len(sys.argv) == 6, 'The number of input parameters must be equal to 5'
        get_bin_info(file_path, info_name, width, height)
    elif file_type == 'png':
        assert len(sys.argv) == 4, 'The number of input parameters must be equal to 3'
        get_png_info(res, info_name)