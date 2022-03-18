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
import numpy as np
import cv2


# rcf preprocess
def preprocess(args):
    """[rcf preprocess]

    Args:
        args ([argparse]): [rcf preprocess parameters]
    """
    src_dir = args.src_dir
    in_files = []

    image_names = os.listdir(src_dir)
    for i in range(len(image_names)):
        if image_names[i].endswith(('jpg', 'png', 'jpeg', 'bmp')):
            comp_name = os.path.join(src_dir, image_names[i])
            in_files.append(comp_name)
    
    i = 0
    h_list, w_list = args.height, args.width # if args.height is list, assign args.height to h_list
    for k in range(len(h_list)):
        h, w = h_list[k], w_list[k]
        save_path = args.save_name + '_{}x{}'.format(h, w)
        os.system('rm -rf {}'.format(save_path))
        os.system('mkdir -p {}'.format(save_path))
    for file in in_files:
        if not os.path.isdir(file):
            i = i + 1
            print(file, "====", i)
            image = cv2.imread(file).astype(np.float32)
            h, w, c = image.shape
            image -= np.array((104.00698793, 116.66876762, 122.67891434))
            image = np.transpose(image, (2, 0, 1))

            save_path = args.save_name + '_{}x{}'.format(h, w)
            temp_name = file[file.rfind('/') + 1:]
            image.tofile(os.path.join(save_path, temp_name.split('.')[0] + ".bin"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing of RCF model')
    parser.add_argument('--src_dir', default='data/BSR/BSDS500/data/images/test', type=str, 
    help='The file records the pictures that need to be preprocessed')
    parser.add_argument('--save_name', default='data/images_bin', type=str, help='Output path, If not exist, create it')
    parser.add_argument('--height', nargs='+',
                        type=int, help='input height')
    parser.add_argument('--width', nargs='+',
                        type=int, help='input width')
    args = parser.parse_args()
    preprocess(args)
