# Copyright 2022 Huawei Technologies Co., Ltd
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

import argparse
import glob
import os
import numpy as np
from pprint import pprint
from PIL import Image

def resize(img, size, interpolation=Image.BILINEAR):
    
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1],interpolation)

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

def deepmar_onnx(in_file):

    input_size = (256, 256)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = Image.open(in_file).convert('RGB')
    img = resize(img, input_size)
    img = np.array(img, dtype=np.float32)
    img = center_crop(img, 224, 224)
    img = img / 255.
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    img[..., 0] /= std[0]
    img[..., 1] /= std[1]
    img[..., 2] /= std[2]

    img = img.transpose(2, 0, 1) # HWC -> CHW
    return img

class GlobDataLoader():
    def __init__(self, glob_pattern, limit=None):
        self.glob_pattern = glob_pattern
        self.limit = limit
        self.file_list = self.get_file_list()
        self.cur_index = 0

    def get_file_list(self):
        return glob.iglob(self.glob_pattern)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_index == self.limit:
            raise StopIteration()
        label = None
        file_path = next(self.file_list)
        with open(file_path, 'rb') as fd:
            data = fd.read()

        self.cur_index += 1
        return get_file_name(file_path), label, data

def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path.rstrip('/')))[0]

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('glob', help='img pth glob pattern.')
    parser.add_argument('result_file', help='result file')
    return parser.parse_args()

def main():

    args = parse_args()
    result_fname = get_file_name(args.result_file)
    pred_result_file = f"{result_fname}/"
    dataset = GlobDataLoader(args.glob+"/*", limit=50000)
    # start preprocess
    for name, _, data in dataset:
        file_path = os.path.join(args.glob, name+'.JPEG')
        img = deepmar_onnx(file_path)
        img.tofile(os.path.join(pred_result_file, name+'.bin'))

    print("success in preprocess")

if __name__ == "__main__":
    main()
