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
import argparse
import config
from PIL import Image
import numpy as np
import cv2


def preprocess(args, side_h=998, side_w=729):
    src_dir = args.src_dir
    in_files = []

    image_names = os.listdir(src_dir)
    for i in range(len(image_names)):
        comp_name = os.path.join(src_dir, image_names[i])
        in_files.append(comp_name)
     
    i = 0
    for file in in_files:
        if not os.path.isdir(file):
            i = i + 1
            print(file, "====", i)
            image = cv2.imread(file)

            h, w= image.shape[:2]
            rescale_fac = max(h, w) / 1000
            if rescale_fac > 1.0:
                h = int(h / rescale_fac)
                w = int(w / rescale_fac)
            
            # Determined according to the minimum distance
            distance_list = []
            for n in range(config.center_len):
                distance_list.append(np.linalg.norm(np.array([h, w])-np.array(config.center_list[n])))
            min_distance = min(distance_list)
            side_h = config.center_list[distance_list.index(min_distance)][0]
            side_w = config.center_list[distance_list.index(min_distance)][1]

            save_path=args.save_path + '_{}x{}'.format(side_h, side_w)

            height, width = image.shape[:2]
            scale_h = side_h / height
            scale_w = side_w / width
            scale = min(scale_h, scale_w) # Calculation of scale with different width and height
            width_scaled = int(width * scale)
            height_scaled = int(height * scale)
            image_scaled = cv2.resize(image, (width_scaled, height_scaled))
            image_array = image_scaled.astype(np.float32)
            image_padded = np.full([side_h, side_w, 3], 0, dtype=np.float32)
            width_offset = (side_w - width_scaled) // 2
            height_offset = (side_h - height_scaled) // 2
            image_padded[height_offset:height_offset + height_scaled, width_offset:width_offset + width_scaled, :] \
                = image_array
            image_norm = image_padded - [123.68, 116.779, 103.939]
            image_norm = np.transpose(image_norm, (2, 0, 1)).astype(np.float32)

            temp_name = file[file.rfind('/') + 1:]
            image_norm.tofile(os.path.join(save_path, temp_name.split('.')[0] + ".bin"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing of CTPN model')
    parser.add_argument('--src_dir', default='data/Challenge2_Test_Task12_Images', type=str, 
    help='The file records the pictures that need to be preprocessed')
    parser.add_argument('--save_path', default='data/images_bin', type=str, help='Output path, If not exist, create it')
    args = parser.parse_args()

    preprocess(args)
