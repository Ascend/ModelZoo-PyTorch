# Copyright 2022 Huawei Technologies Co., Ltd
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
from PIL import Image
import numpy as np
import multiprocessing


data_config = {
    'resize': 300,
    'mean': [127, 127, 127],
    'std': 128.0
}


def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        return img.resize((size, size), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def gen_input_bin(src_path, file_batches, batch, save_path):
    for file in file_batches[batch]:
        # RGBA to RGB
        image = Image.open(os.path.join(src_path, file)).convert('RGB')
        image = resize(image, data_config['resize'])
        img = np.array(image, dtype=np.float32)
        
        img -= np.array(data_config['mean'], dtype=np.float32)
        img /= np.array(data_config['std'], dtype=np.float32)
        img = img.transpose(2, 0, 1) # ToTensor: HWC -> CHW
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


def preprocess(src_path, save_path):
    files = os.listdir(src_path)

    file_batches = [
        files[i:i + 500] for i in range(0, 50000, 500) 
        if files[i:i + 500] != []
    ]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(
                                src_path, file_batches, batch, save_path))
    thread_pool.close()
    thread_pool.join()
    print("Done! please ensure preprocess succcess.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('data preprocess.')
    parser.add_argument('--src_path', type=str, required=True, 
                        help='path to original dataset.')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='a directory to save bin files.')
    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    preprocess(args.src_path, args.save_path)
