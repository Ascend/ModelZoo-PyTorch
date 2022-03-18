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

"""
This script converts the 'bin' file to 'PNG' file.
"""

import os
import numpy as np
import torch
import argparse
from torchvision.utils import save_image


def post_process(args):
    img_bin_list = os.listdir(args.benchmark_result_path)
    img_bin_list.sort()
    img_list_len = len(img_bin_list)
    shape = (1, 1, 32, 32)
    for i in range(img_list_len):
        source_path = os.path.join(args.benchmark_result_path, img_bin_list[i])
        base_name = os.path.basename(img_bin_list[i])[:-4]
        target_path = os.path.join(args.save_path, base_name + ".png")
        print(source_path, " ===> ", target_path)
        img = np.fromfile(source_path, dtype=np.float32)
        img = torch.from_numpy(img)
        img = img.view(shape)
        save_image(img, normalize=True, nrow=1, fp=target_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_result_path", default="./result_bs1/dumpOutput_device0", type=str)
    parser.add_argument("--save_path", default="./postprocess_img/", type=str)
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    post_process(args)


if __name__ == "__main__":
    main()
