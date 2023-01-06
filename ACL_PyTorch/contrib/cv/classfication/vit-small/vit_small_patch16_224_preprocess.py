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

import sys
sys.path.append('./pytorch-image-models')
from tqdm import tqdm
import os
from timm.data import create_loader, ImageDataset


os.environ['device'] = 'cpu'


def preprocess(src_path, save_path):
    with open("vit_prep_bin.info", "w") as f:
        loader = create_loader(
            ImageDataset(src_path),
            input_size=(3, 224, 224),
            batch_size=64,
            is_training=False,
            use_prefetcher=True,
            interpolation="bicubic",
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_workers=8,
            crop_pct=0.9)

        for batch_idx, (input, target, path) in tqdm(enumerate(loader), total=len(loader)):
            base_index = batch_idx * 64
            for idx, (img, p) in enumerate(zip(input, path)):
                index = base_index + idx
                filename = os.path.basename(p)
                img = img.numpy()
                save_name = os.path.join(
                    save_path, filename.split('.')[0] + ".bin")
                img.tofile(save_name)
                info = "{0} {1} 224 224\n".format(index, save_name)
                f.write(info)


if __name__ == '__main__':
    os.makedirs(sys.argv[2], exist_ok=True)
    preprocess(sys.argv[1], sys.argv[2])
