# Copyright 2020 Huawei Technologies Co., Ltd
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
import sys
import argparse
import multiprocessing
import tqdm
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")

from glob import glob
from PIL import Image
from tools.utils import build_transforms

fix_shape = 1344
trans = build_transforms(fix_shape)


def gen_input_bin(file_batches, batch):

    for file in tqdm.tqdm(file_batches[batch]):
       

        image = Image.open(os.path.join(flags.image_src_path, file)).convert('RGB')
        dummy_input = trans(image)

        dummy_input.tofile(os.path.join(flags.bin_file_path, file.split('.')[0] + ".bin"))


def preprocess(src_path):
    files = os.listdir(src_path)
    file_batches = [files[i:i + 100] for i in range(0, 5000, 100) if files[i:i + 100] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")


def get_bin_info(file_path, info_name, width, height):
    bin_images = glob(os.path.join(file_path, '*.bin'))
    with open(info_name, 'w') as file:
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, width, height])
            file.write(content)
            file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of MaskRCNN PyTorch model')
    parser.add_argument("--image_src_path", default="/opt/npu/coco/val2017/", help='image of dataset')
    parser.add_argument("--bin_file_path", default="./bins/", help='Preprocessed image buffer')
    parser.add_argument("--bin_info_name", default="retinamask_coco2017.info")
    parser.add_argument("--input_size", default='1344', type=str, help='input tensor size')
    flags = parser.parse_args()
    if not os.path.exists(flags.bin_file_path):
        os.makedirs(flags.bin_file_path)
    preprocess(flags.image_src_path)

    # gen bins_info
    get_bin_info(flags.bin_file_path, flags.bin_info_name, flags.input_size, flags.input_size)
