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
from PIL import Image
import argparse
import numpy as np
import torchvision.transforms as T
from config import cfg

def main(args):
    transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            ])

    src_path = os.path.join(args.data_root_path, args.src_path)   
    save_path = os.path.join(args.save_root_path, args.save_path)
    in_files = os.listdir(src_path)
    for file in enumerate(in_files):
        input_image = Image.open(src_path + '/' + file).convert('RGB')
        input_tensor = transform(input_image)
        img = np.array(input_tensor).astype(np.float32)
        
        save_file_path = os.path.join(save_path, file.split('.')[0] + ".bin")
        img.tofile(save_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data preprocess")
    parser.add_argument("--data_root_path", default="/opt/npu/", help="data root path", type=str)
    parser.add_argument("--src_path", default="DukeMTMC-reID/bounding_box_test", help="src path", type=str)
    parser.add_argument("--save_root_path", default="/home/wxq/", help="save root path", type=str)
    parser.add_argument("--save_path", default="DukeMTMC-reID/bin_data/gallery", help="save path", type=str)
    args = parser.parse_args()
    main(args)