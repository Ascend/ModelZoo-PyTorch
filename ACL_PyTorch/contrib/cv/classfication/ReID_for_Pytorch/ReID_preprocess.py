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
from torchvision import transforms
import multiprocessing
from tqdm import tqdm

preprocess = transforms.Compose([
        transforms.Resize([256, 128]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def gen_input_bin(file_batches, batch):
    i = 0
    for file in file_batches[batch]:
        if ".db" in file:
            continue

        input_image = Image.open(os.path.join(src_path, file)).convert('RGB')
        input_tensor = preprocess(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))

def ReID_preprocess(src_path, save_path):
    files = os.listdir(src_path)
    file_batches = [files[i:i + 500] for i in range(0, 50000, 500) if files[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    pbar = tqdm(range(len(file_batches)))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch),
                                callback=lambda _: pbar.update(1),
                                error_callback=lambda _: pbar.update(1))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")

if __name__ == '__main__':
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    ReID_preprocess(src_path, save_path)
