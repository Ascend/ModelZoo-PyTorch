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
import numpy as np
import parse


def om_pre(ar):
    ar.batch_size = 1

    if not os.path.exists(ar.bin_dir):
        os.makedirs(ar.bin_dir)
    data, _, label = parse.load_data_model(ar)
    with open('albert.info', 'w') as f:
        label2 = np.array([i.numpy() for i in label]).flatten()
        np.savetxt('albert.label', label2)
        for idx, dat in enumerate(data):
            id, mask, token = dat['input_ids'], dat['attention_mask'], dat['token_type_ids']
            id.numpy().tofile('{}/input_ids_{}.bin'.format(ar.bin_dir, idx))
            f.write("{} {}input_ids_{}.bin\n".format(idx, ar.bin_dir, idx))

            mask.numpy().tofile('{}/attention_mask_{}.bin'.format(ar.bin_dir, idx))
            f.write("{} {}attention_mask_{}.bin\n".format(
                idx, ar.bin_dir, idx))

            token.numpy().tofile('{}/token_type_ids_{}.bin'.format(ar.bin_dir, idx))
            f.write("{} {}token_type_ids_{}.bin\n".format(
                idx, ar.bin_dir, idx))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_dir", type=str, default='./albert_pytorch/outputs/SST-2/',
                        help="dir of pth, load args.bin and model.bin")
    parser.add_argument("--bin_dir", type=str, default='bin/',
                        help="dir of bin, info, label")
    ar = parser.parse_args()

    ar.pth_arg_path = ar.pth_dir + "training_args.bin"
    ar.data_type = 'dev'
    ar.data_path = './albert_pytorch/dataset/SST-2'

    om_pre(ar)


if __name__ == "__main__":
    main()
