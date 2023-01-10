# Copyright 2023 Huawei Technologies Co., Ltd
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
from timm.data import create_loader, ImageDataset
import os
import numpy as np
import argparse

os.environ['device'] = 'cpu'

def preprocess_volo(data_dir, save_path, batch_size):
    f = open("volo_val_bs"+str(batch_size)+".txt", "w")

    loader = create_loader(
        ImageDataset(data_dir),
        input_size=(3, 224, 224),
        batch_size=batch_size,
        use_prefetcher=False,
        interpolation="bicubic",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_workers=4,
        crop_pct=0.96,
        pin_memory=False,
        tf_preprocessing=False)

    for batch_idx, (input, target) in enumerate(loader):
        img = np.array(input).astype(np.float32)
        if img.shape[0] < batch_size:
            continue
        save_name = os.path.join(save_path, "test_" + str(batch_idx) + ".bin")
        print(save_name)
        img.tofile(save_name)
        info = "%s " % ("test_" + str(batch_idx) + ".bin")
        for i in range(batch_size):
            info = info + str(int(target[i])) + " "
        info = info + "\n"
        f.write(info)

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Imagenet val_dataset preprocess')
    parser.add_argument('--src', type=str, default='./',
                        help='imagenet val dir.')
    parser.add_argument('--des', type=str, default='./',
                        help='preprocess dataset dir.')
    parser.add_argument('--batchsize', type=int, default='1',
                        help='batchsize.')
    args = parser.parse_args()
    src = args.src
    des = args.des
    bs = args.batchsize
    files = None
    if not os.path.exists(src):
        print('this path not exist')
        exit(0)
    os.makedirs(des, exist_ok=True)
    preprocess_volo(src, des, bs)

    # python volo_224_preprocess.py --src /opt/npu/val --des /opt/npu/data_bs1 --batchsize 1
