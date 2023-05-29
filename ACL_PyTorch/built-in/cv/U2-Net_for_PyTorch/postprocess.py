# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import glob
import argparse

from PIL import Image
from skimage import io
import numpy as np
from tqdm import tqdm


WORK_DIR = './workspace/U-2-Net'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Postprocess for U-2-Net'
    )
    parser.add_argument('--image_dir', type=str, default='./datasets/ECSSD/images',
                        help='input dataset image dir')
    parser.add_argument('--save_dir', type=str, default='./test_vis_ECSSD')
    parser.add_argument('--out_dir', type=str, default='./result/dumpOutput_device0')
    global_args = parser.parse_args()
    return global_args


def save_output(image_name, predict_np, d_dir):
    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    save_output_result = os.path.join(d_dir, imidx + '.png')
    imo.save(save_output_result)


def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def postprocess(ori_image_list, bin_dir, num, save_dir):
    for idx in tqdm(range(num)):
        bin_path = os.path.join(bin_dir, '{}_0.bin'.format(idx))
        bin_data = np.fromfile(bin_path, dtype=np.float32).reshape([320, 320])
        bin_data = normPRED(bin_data)
        image_path = ori_image_list[idx]
        save_output(image_path, bin_data, save_dir)


if __name__ == '__main__':
    args = parse_args()
    out_dir = args.out_dir
    save_dir_path = args.save_dir
    image_dir = args.image_dir
    bin_files = os.listdir(out_dir)
    os.makedirs(save_dir_path, exist_ok=True)
    img_name_list = glob.glob(image_dir + os.sep + '*')
    img_name_list.sort()
    postprocess(img_name_list, out_dir, len(bin_files), save_dir_path)