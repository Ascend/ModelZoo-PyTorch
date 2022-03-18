# encoding=utf-8
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

from __future__ import absolute_import, division, print_function

import argparse
import os
import parse


def om_pre(ar):
    ar.batch_size = 1
    bin_path = ar.data_path + '/bin'
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    data, _, label = parse.load_data_model(ar)
    print('data num: %d' % len(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_dir", type=str, default='./albert_pytorch/outputs/SST-2/',
                        help="dir of pth, load args.bin and model.bin")
    ar = parser.parse_args()

    ar.pth_arg_path = ar.pth_dir + "training_args.bin"
    ar.data_type = 'dev'
    ar.data_path = './albert_pytorch/dataset/SST-2'

    om_pre(ar)


if __name__ == "__main__":
    main()
