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

import os
import argparse
import numpy as np


def om_post(ar):
    res_dir=ar.result_dir
    labels = np.load(ar.label_path)
    yes = 0
    for i in range(len(labels)):
        res_path = os.path.join(res_dir, f'{i}_0.npy')
        res = np.load(res_path).flatten()
        label = labels.flatten()[i]
        if (res[1] > res[0] and label > 0.5) or (res[1] < res[0] and label < 0.5):
            yes += 1
    print("acc = {:.3f}".format(yes / len(labels)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="", type=str, required=True,
                        help="infer result dir.")
    parser.add_argument("--label_path", default="", type=str, required=True,
                        help="path for gt label.")
    ar = parser.parse_args()

    om_post(ar)


if __name__ == "__main__":
    main()
