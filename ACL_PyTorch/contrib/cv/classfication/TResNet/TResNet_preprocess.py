#!/usr/bin/env python3
# Copyright 2020 Huawei Technologies Co., Ltd
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
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.append(r"./pytorch-image-models")
from timm.data import create_loader, ImageDataset
os.environ['device'] = 'cpu'


def preprocess(src_path, save_path):
    f = open("tresnet_prep_bin.info", "w")
    loader = create_loader(
        ImageDataset(src_path),
        input_size=(3, 224, 224),
        batch_size=64,
        is_training=False,
        use_prefetcher=True,
        interpolation="bilinear",
        mean=(0, 0, 0),
        std=(1, 1, 1),
        num_workers=8,
        crop_pct=0.875)


    for batch_idx, (input, target, path) in enumerate(loader):
        base_index = batch_idx * 64
        for idx, (img, p) in enumerate(zip(input, path)):
            index = base_index + idx
            filename = os.path.basename(p)
            print(filename, "===", index)
            img = np.array(img).astype(np.float32)
            save_name = os.path.join(save_path, filename.split('.')[0] + ".bin")
            img.tofile(save_name)
            info = "%d %s 224 224\n" % (index, save_name)
            f.write(info)
    f.close()


if __name__ == '__main__':
    imagenet_path = sys.argv[1] #/opt/npu/imagenet/val
    output_path = sys.argv[2] #"./prep_dataset/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    preprocess(imagenet_path, output_path)
