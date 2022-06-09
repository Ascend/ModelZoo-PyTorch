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
import mmcv
import numpy as np
import argparse

dataset_config = {
    'resize': (640, 640),
    'mean': [103.530, 116.280, 123.675],
    'std': [1, 1, 1],
}
def preprocess(img_src_path, save_path, batch_size):
    image_list = ["000000455085.jpg", "000000570756.jpg", "000000446651.jpg", "000000372317.jpg",
                  "000000491216.jpg", "000000524742.jpg", "000000465585.jpg", "000000015751.jpg",
                  "000000511384.jpg", "000000434297.jpg", "000000219283.jpg", "000000473869.jpg",
                  "000000231822.jpg", "000000292225.jpg", "000000142971.jpg", "000000113720.jpg"]
    for i, file in enumerate(image_list):
        image = mmcv.imread(os.path.join(img_src_path, file), backend='cv2')
        image, scalar = mmcv.imrescale(image, (args.model_input_height, args.model_input_width), return_scale=True)
        mean = np.array(dataset_config['mean'], dtype=np.float32)
        std = np.array(dataset_config['std'], dtype=np.float32)
        image = mmcv.imnormalize(image, mean, std, to_rgb=False)
        image = mmcv.impad(image, shape=(args.model_input_height, args.model_input_width),
                           pad_val=(args.model_pad_val, args.model_pad_val, args.model_pad_val))
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)
        input_data = np.expand_dims(image, axis=0)
        print("shape:", input_data.shape)

        if i % batch_size == 0:
            output_data = input_data
        else:
            output_data = np.concatenate((output_data, input_data), axis=0)

        if (i + 1) % batch_size == 0:
            output_data.tofile("{}/quant.bin".format(save_path))
            break


if __name__ == "__main__":
    """
    python3 generate_data.py \
            --img_info_file=img_info_amct.txt \
            --save_path=amct_data \
            --batch_size=1
    """
    parser = argparse.ArgumentParser(description='yolox_tiny generate quant data')
    parser.add_argument('--img_src_path', type=str, default="root/data/coco", help='original data')
    parser.add_argument('--save_path', type=str, default="./int8data", help='data for amct')
    parser.add_argument('--batch_size', type=int, default=16, help='om batch size')
    parser.add_argument('--model_input_width', type=int, default=640, help='image width')
    parser.add_argument('--model_input_height', type=int, default=640, help='image height')
    parser.add_argument('--model_pad_val', type=int, default=0, help='image height')
    args = parser.parse_args()
    preprocess(args.img_src_path, args.save_path, args.batch_size)
