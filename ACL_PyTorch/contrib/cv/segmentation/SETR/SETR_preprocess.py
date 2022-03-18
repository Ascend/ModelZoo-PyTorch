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
'''
    将val输入图像分割出8个768*768的图片，并生成*.bin文件
'''

import os
import sys
import numpy as np
from mmcv.utils import Config
from mmseg.datasets import build_dataloader, build_dataset

def gen_data_loader(distributed, cfg):

    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
    return val_dataloader


def gen_input_data_768X768(data_loader, out_dir, test_cfg):

    for i, data in enumerate(data_loader):
        file_name = data['img_metas'][0].data[0][0]['ori_filename']
        img = data['img'][0] # tensor 1,3,1025, 2049

        h_stride, w_stride = test_cfg.stride
        h_crop, w_crop = test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        index = 0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                crop_img = np.array(crop_img).astype(np.float32)
                output_path = os.path.join(out_dir, "/".join(file_name.split('/')[:-1]))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                crop_img_name = "{}[{}].bin".format(file_name.split('.')[0], index)
                crop_img.tofile(os.path.join(out_dir, crop_img_name))
                index += 1

if __name__ == '__main__':
    model_config_file = sys.argv[1]
    output_file = sys.argv[2]

    cfg = Config.fromfile(model_config_file)
    dataloader = gen_data_loader(False, cfg)
    gen_input_data_768X768(dataloader, output_file, cfg.test_cfg)