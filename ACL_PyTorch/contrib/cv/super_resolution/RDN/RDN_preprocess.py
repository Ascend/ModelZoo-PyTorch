# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import mmcv
import numpy as np
import PIL.Image as pil_image
import cv2

def resize(img, size):
    org_h = img.shape[0]
    org_w = img.shape[1]
    scale_ratio = min(size[0] / org_w, size[1] / org_h)
    new_w = int(np.floor(org_w * scale_ratio))
    new_h = int(np.floor(org_h * scale_ratio))
    resized_img = mmcv.imresize(img, (new_w, new_h), backend='cv2')
    return resized_img

def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(os.path.realpath(path))

def preprocess(src_path, save_path):
    in_files = os.listdir(src_path)
    for file in in_files:
        image = cv2.imread(os.path.join(src_path, file))
        image_input_width = 228
        image_input_height = 228
        image = resize(image, [image_input_width, image_input_height])

        rh = image.shape[0]
        rw = image.shape[1]
        pad_left = (image_input_width - rw) // 2
        pad_top = (image_input_height - rh) // 2
        pad_right = image_input_width - pad_left - rw
        pad_bottom = image_input_height - pad_top - rh
        image = mmcv.impad(image, padding=(pad_left, pad_top, pad_right, pad_bottom), pad_val=0)
        
        # CV2转PIL
        image = pil_image.fromarray(image)
        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
        hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)

        lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        data_path = os.path.join(save_path, 'data')
        label_path = os.path.join(save_path, 'label')
        mkdir(data_path)
        mkdir(label_path)
        lr.tofile(os.path.join(data_path, file.split('.')[0] + ".bin"))
        np.save(os.path.join(label_path, file.split('.')[0]), hr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    mkdir(args.save_path)
    preprocess(args.src_path, args.save_path)
