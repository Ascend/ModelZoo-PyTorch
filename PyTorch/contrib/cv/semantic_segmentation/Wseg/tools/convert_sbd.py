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

"""Convert .mat segmentation mask of SBD to .png
See: https://github.com/visinf/1-stage-wseg/issues/9
"""

import os
import sys
import glob
import argparse
from PIL import Image
from scipy.io import loadmat

# load tqdm optionally
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def args():
    parser = argparse.ArgumentParser(description="Convert SBD .mat to .png")
    parser.add_argument("--inp", type=str, default='./dataset/cls/',
                        help="Directory with .mat files")
    parser.add_argument("--out", type=str, default='./dataset/cls_png/',
                        help="Directory where to save .png files")
    return parser.parse_args(sys.argv[1:])


def convert(opts):

    # searching for files .mat
    opts.inp = opts.inp + ("" if opts.inp[-1] == "/" else "/")
    filelist = glob.glob(opts.inp + "*.mat")
    print("Found {:d} files".format(len(filelist)))

    if len(filelist) == 0:
        return

    # check output directory
    if not os.path.isdir(opts.out):
        print("Creating {}".format(opts.out))
        os.makedirs(opts.out)


    for filepath in tqdm(filelist):

        x = loadmat(filepath)
        y = x['GTcls']['Segmentation'][0][0]

        # converting to PIL image
        png = Image.fromarray(y)

        name = os.path.basename(filepath).replace(".mat", ".png")
        png.save(os.path.join(opts.out, name))


if __name__ == "__main__":
    opts = args()
    convert(opts)
