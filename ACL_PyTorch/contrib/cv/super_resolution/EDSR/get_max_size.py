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

import imageio
import os
import argparse

parser = argparse.ArgumentParser(description='get max size of images in dir')
parser.add_argument('--dir', default='/root/datasets/div2k/LR', type=str, metavar='PATH',
                    help='png dir path')
args = parser.parse_args()

if __name__ == '__main__':
    max_size = 0
    for file in os.listdir(args.dir):
        image = imageio.imread(os.path.join(args.dir, file))
        x, y, z = image.shape
        if max_size < max(x, y):
            max_size = max(x, y)

    print("the max size of images is", max_size)
