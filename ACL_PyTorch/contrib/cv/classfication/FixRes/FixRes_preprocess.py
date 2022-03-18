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

import os
import sys
import numpy as np
from PIL import Image
import multiprocessing
from FixRes.imnet_evaluate.transforms import get_transforms
import argparse

transformation=get_transforms(input_size=384, test_size=384, kind='full',\
    crop=True, need=('train', 'val'), backbone=None)
preprocess = transformation['val']

def gen_input_bin(file_batches, batch, src_path, save_path):
    """Generate input bin files

    Args:
        file_batches ([str]): batches of files
        batch (int): batch index
    """
    i = 0
    for filename in file_batches[batch]:
        if ".db" in filename:
            continue
        i = i + 1
        print("batch", batch, filename, "===", i)

        input_image = Image.open(os.path.join(src_path, filename)).convert('RGB')
        if '/' in filename:
            _, imgname = filename.split('/')
        else:
            imgname = filename
        if imgname.endswith('.JPEG'):
            imgname = imgname.strip('.JPEG')
        elif imgname.endswith('.jpeg'):
            imgname = imgname.strip('.jpeg')
        else:
            raise ValueError('Invalid image format')
        input_tensor = preprocess(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, imgname + ".bin"))


def FixRes_preprocess(src_path, save_path):
    """Do preprocess for FixRes

    Args:
        src_path (str): path of imagenet folder
        save_path (str): path to save bin files
    """
    folder_list = os.listdir(src_path)
    if folder_list[0].endswith('.JPEG'):
        # val/xxxx.JPEG
        files = folder_list
    else:
        # val/xxxx/xxxx.JPEG
        files = []
        for folder in folder_list:
            file_list = os.listdir(os.path.join(src_path, folder))
            for filename in file_list:
                files.append(os.path.join(folder, filename))
    file_batches = [files[i:i + 500] for i in range(0, 50000, 500) if files[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch, src_path, save_path))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess script for FixRes \
        model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src-path', default='', type=str, help='path of imagenet')
    parser.add_argument('--save-path', default='', type=str, help='path to save bin files')
    args = parser.parse_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(os.path.realpath(args.save_path))
    FixRes_preprocess(args.src_path, args.save_path)
