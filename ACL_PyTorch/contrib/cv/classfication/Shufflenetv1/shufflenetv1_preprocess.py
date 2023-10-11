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
import sys
import PIL
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
import multiprocessing

def ToBGRTensor(img):
    assert isinstance(img, (np.ndarray, PIL.Image.Image))
    if isinstance(img, PIL.Image.Image):
        img = np.asarray(img)
    img = img[:, :, ::-1] # 2 BGR
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    return img

def OpencvResize(img,size):
    assert isinstance(img, PIL.Image.Image)
    img = np.asarray(img)  # (H,W,3) RGB
    img = img[:, :, ::-1]  # 2 BGR
    img = np.ascontiguousarray(img)
    H, W, _ = img.shape
    target_size = (int(size / H * W + 0.5), size) if H < W else (size, int(size / W * H + 0.5))
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1]  # 2 RGB
    img = np.ascontiguousarray(img)
    img = Image.fromarray(img)
    return img

def gen_input_bin(save_path, file_batches, batch):
    i = 0
    for file in file_batches[batch]:
        i = i + 1
        # RGBA to RGB
        image = Image.open(os.path.join(src_path, file)).convert('RGB')
        image = OpencvResize(image, 256)
        crop = transforms.CenterCrop(224)
        image = crop(image)
        image = ToBGRTensor(image)
        img = np.array(image, dtype=np.uint8)
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))

def preprocess(src_path, save_path):
    files = os.listdir(src_path)
    file_batches = [files[i:i + 500] for i in range(0, 50000, 500) if files[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(save_path, file_batches, batch))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")

if __name__ == "__main__":
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    src_path = os.path.realpath(src_path)
    save_path = os.path.realpath(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    preprocess(src_path, save_path)