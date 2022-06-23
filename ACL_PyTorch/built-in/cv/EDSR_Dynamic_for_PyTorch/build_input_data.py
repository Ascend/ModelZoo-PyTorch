# Copyright 2022 Huawei Technologies Co., Ltd
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


SHAPES = [
    (240, 320),
    (480, 640),
    (720, 1280),
    (1080, 1920),
]
parser = argparse.ArgumentParser(description='EDSR preprocess script')
parser.add_argument('-s', '--save_dir', default='./inputs', type=str, metavar='PATH',
                    help='save dir for generated input data')
args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)


def build_data():
    for shape in SHAPES:
        height, width = shape
        save_path = os.path.join(args.save_dir, "{}_{}.bin".format(height, width))
        input_data = np.random.rand(3, height, width).astype("float16")
        input_data.tofile(save_path)


if __name__ == '__main__':
    build_data()
