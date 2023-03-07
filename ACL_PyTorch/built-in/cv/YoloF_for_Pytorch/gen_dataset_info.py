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
import pickle as pk
import mmcv
from mmdet.datasets import build_dataset


ann_file = '/annotations/instances_val2017.json'
img_prefix = '/val2017/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess of YOLOX PyTorch model')
    parser.add_argument("--image_src_path", default="/opt/npu/coco/val2017", help='image of dataset')
    parser.add_argument("--config_path", default="./configs/yolof/yolof_r50_c5_8x8_1x_coco.py", \
                                                                 help='Preprocessed image buffer')
    parser.add_argument("--bin_path", default="val2017_bin", help='Get image meta')
    parser.add_argument("--meta_path", default="val2017_bin_meta", type=str, help='input tensor height')
    parser.add_argument("--info_name", default="yolof.info", type=str, help='input tensor width')
    parser.add_argument("--info_meta_name", default="yolof_meta.info ", type=str, help='input tensor width')
    parser.add_argument("--width", default=640, type=int, help='image pad value')
    parser.add_argument("--height", default=640, type=int, help='image pad value')
    flags = parser.parse_args()

    image_src_path = flags.image_src_path
    config_path = flags.config_path
    bin_path = flags.bin_path
    meta_path = flags.meta_path
    info_name = flags.info_name
    info_meta_name = flags.info_meta_name
    width = int(flags.width)
    height = int(flags.height)

    cfg = mmcv.Config.fromfile(config_path)
    cfg.data.test.ann_file = image_src_path + ann_file
    cfg.data.test.img_prefix = image_src_path + img_prefix
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)

    with open(info_name, "w") as fp1, open(info_meta_name, "w") as fp2:
        for idx in range(5000):
            img_id = dataset.img_ids[idx]
            fp1.write("{} {}/{:0>12d}.bin {} {}\n".format(idx, bin_path, img_id, width, height))
            fp_meta = open("%s/%012d.pk" % (meta_path, img_id), "rb")
            meta = pk.load(fp_meta)
            fp_meta.close()
            fp2.write("{} {}/{:0>12d}.bin {}\n".format(
                idx,
                meta_path,
                img_id,
                meta['scalar']
            ))
    print("Get info done!")
