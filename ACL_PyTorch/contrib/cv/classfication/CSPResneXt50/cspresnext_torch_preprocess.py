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
import PIL
from PIL import Image
import cv2
import math
import numpy as np
import torch
from torchvision import transforms
import multiprocessing

def gen_input_bin(save_path, file_batches, batch):
    img_size = 224
    crop_pct = 0.875
    scale_size = int(math.floor(img_size / crop_pct))
    i = 0
    for file in file_batches[batch]:
        i = i + 1
        print("batch", batch, file, "===", i)
        input_image = Image.open(os.path.join(src_path, file)).convert('RGB')
        preprocessor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
        input_tensor = preprocessor(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


def preprocess(src_path, save_path):
    files = os.listdir(src_path)
    file_batches = [files[i:i + 500] for i in range(0, 50000, 500) if files[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(save_path, file_batches, batch))
    thread_pool.close()
    thread_pool.join()



if __name__=="__main__":
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    src_path = os.path.realpath(src_path)
    save_path = os.path.realpath(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    preprocess(src_path, save_path)
