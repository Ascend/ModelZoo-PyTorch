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
from PIL import Image
import numpy as np
import multiprocessing
from torchvision import transforms
from tqdm import tqdm


def gen_input_bin(save_path, file_batches, batch):
    for file in file_batches[batch]:
        # RGBA to RGB
        image = Image.open(os.path.join(src_path, file)).convert('RGB')
        resize = transforms.Resize(256)
        image = resize(image)
        crop = transforms.CenterCrop(224)
        image = crop(image)
        tt = transforms.ToTensor()
        image = tt(image)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        image = np.array(image, dtype=np.float32)
        image.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


def preprocess(src_path, save_path):
    files = os.listdir(src_path)
    file_batches = [files[i:i + 500]
                    for i in range(0, 50000, 500) if files[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    pbar = tqdm(range(len(file_batches)))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin,
                                args=(save_path, file_batches, batch),
                                callback=lambda _: pbar.update(1),
                                error_callback=lambda _: pbar.update(1))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")


if __name__ == '__main__':
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    src_path = os.path.realpath(src_path)
    save_path = os.path.realpath(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    preprocess(src_path, save_path)
