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

import sys
import os
import multiprocessing

from PIL import Image
import torch
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
from tqdm import tqdm

def gen_input_bin(file_batches, batch, input_dir, output_dir):
    """
    Data augmentation and save data to binary files.
    """
    for file_name in file_batches[batch]:
        pilimg = Image.open(os.path.join(input_dir, file_name))
        pilimg = pilimg.convert("RGB")
        val_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])
        img_tensor = val_transformer(pilimg)
        img_tensor = torch.unsqueeze(img_tensor, dim=0).float()
        img_tensor.reshape(1, 3, 224, 224)
        img_numpy = img_tensor.cpu().numpy()
        img_numpy.tofile(os.path.join(output_dir, file_name.split('.')[0] + ".bin"))


def preprocess(input_dir, output_dir):
    """
    Preprocess data with multiprocess.
    """
    file_names = os.listdir(input_dir)
    file_batches = [file_names[i:i+500] for i in range(0, 50000, 500) if file_names[i:i+500] != []]

    pbar = tqdm(total=len(file_batches))
    pbar.set_description("Preprocessing")
    update = lambda *args:pbar.update()
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch, input_dir, output_dir), callback=update)
    thread_pool.close()
    thread_pool.join()
    print("Except will not report in thread! Please ensure bin file_names generated.")

if __name__ == "__main__":
    data_dir = sys.argv[1]
    save_dir = sys.argv[2]
    if not os.path.isdir(save_dir):
        os.makedirs(os.path.realpath(save_dir))
    preprocess(data_dir, save_dir)
