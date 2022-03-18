# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
This script converts the 'bin' file to 'JPG' file.
"""

import os
import torch
import argparse
import numpy as np

from torchvision.utils import save_image

##############################################################################
# Function
##############################################################################
def post_process(args):
    sub_path = os.path.join(args.save_path, "img" + "_bs" + str(args.batch_size))
    if not os.path.exists(sub_path) and args.save_img:
        os.makedirs(sub_path)

    result_path_list = os.listdir(args.result_path)
    result_path_list.sort(reverse=True)
    bin_path_name = result_path_list[0]
    bin_path = os.path.join(args.result_path, bin_path_name)
    if os.path.isfile(bin_path):
        bin_path = args.result_path
    else:
        print("==>There are subfolders in the given folder.",
              "==>The subfolder with the latest generation date is selected by default.", sep='\n')
    img_bin_list = os.listdir(bin_path)
    img_bin_list.sort()
    shape = (-1, 3, 128, 128)
    x = []
    num_count = 0
    for i in range(len(img_bin_list)):
        source_path = os.path.join(bin_path, img_bin_list[i])
        img = np.fromfile(source_path, dtype=np.float32)
        img = torch.from_numpy(img)
        img = img.view(shape)
        if args.save_npz:
            x += [img.cpu().numpy()]
        batch, _, _, _ = img.shape
        for j in range(batch):
            if args.save_img:
                base_name = os.path.basename(img_bin_list[i])[:-4] + "_" + str(i*args.batch_size + j)
                target_path = os.path.join(sub_path, base_name + ".jpg")
                save_image(img[j], normalize=True, nrow=1, fp=target_path)
            num_count += 1

    if args.save_npz:
        x = np.concatenate(x, 0)[:num_count]
        img_npz_filename = './gen_img' + '_bs' + str(args.batch_size) + '.npz'
        np.savez(img_npz_filename, **{'x': x})

    print("==> Converting successfully.")


##############################################################################
# Main
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, default="./outputs_bs1_om")
    parser.add_argument("--save-path", type=str, default="./postprocess_img")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-img", action='store_true')
    parser.add_argument("--save-npz", action='store_true')
    opt = parser.parse_args()

    if not os.path.exists(opt.save_path) and opt.save_img:
        os.makedirs(opt.save_path)

    if opt.save_img or opt.save_npz:
        post_process(opt)
    else:
        print("后处理需要指定--save-img和--save-npz中的一个或者全部")
