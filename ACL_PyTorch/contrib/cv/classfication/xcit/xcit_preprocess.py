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

import sys

sys.path.append(r'./xcit')
from datasets import build_transform
from main import get_args_parser
import argparse
import os
import numpy as np
from PIL import Image


def preprocess(args):
    Transform = build_transform(is_train=False, args=args)
    val_path = os.path.join(args.data_path,'val')
    save_path = os.path.join(args.resume)
    val_files = os.listdir(val_path)
    i = 0
    for val_set in val_files:
        valset_p = os.path.join(val_path, val_set)
        if not os.path.isdir(valset_p):
            i = i + 1
            file = val_set
            print(file, "===", i)
            input_image = Image.open(valset_p).convert('RGB')
            input_tensor = Transform(input_image)
            img = np.array(input_tensor).astype(np.float32)
            img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))
            continue
        files = os.listdir(valset_p)
        for file in files:
            i = i + 1
            print(file, "===", i)
            input_image = Image.open(valset_p + '/' + file).convert('RGB')
            input_tensor = Transform(input_image)
            img = np.array(input_tensor).astype(np.float32)
            img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    preprocess(args)
