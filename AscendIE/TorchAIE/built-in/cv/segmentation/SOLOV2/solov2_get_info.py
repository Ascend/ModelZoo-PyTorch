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
import mmcv
import pickle as pk
from mmdet.datasets import build_dataset


def load_config(config_path, image_src_path, ann_file, img_prefix):
    cfg = mmcv.Config.fromfile(config_path)
    cfg.data.test.ann_file = image_src_path + ann_file
    cfg.data.test.img_prefix = image_src_path + img_prefix
    return cfg


def write_info(dataset, bin_path, width, height, info_name):
    with open(info_name, "w") as fp1:
        for idx in range(5000):
            img_id = dataset.img_ids[idx]
            fp1.write("{} {}/{:0>12d}.bin {} {}\n".format(idx, bin_path, img_id, width, height))


def write_meta_info(dataset, meta_path, info_meta_name):
    with open(info_meta_name, "w") as fp2:
        for idx in range(5000):
            img_id = dataset.img_ids[idx]
            with open("%s/%012d.pk" % (meta_path, img_id), "rb") as fp_meta:
                meta = pk.load(fp_meta)
                fp2.write("{} {}/{:0>12d}.bin {} {} {} {}\n".format(
                    idx,
                    meta_path,
                    img_id,
                    meta['img_shape'][1],
                    meta['img_shape'][0],
                    meta['ori_shape'][1],
                    meta['ori_shape'][0]
                ))


def main():
    image_src_path = sys.argv[1]
    config_path = sys.argv[2]
    bin_path = sys.argv[3]
    meta_path = sys.argv[4]
    info_name = sys.argv[5]
    info_meta_name = sys.argv[6]
    width = int(sys.argv[7])
    height = int(sys.argv[8])

    ann_file = '/annotations/instances_val2017.json'
    img_prefix = '/val2017/'

    cfg = load_config(config_path, image_src_path, ann_file, img_prefix)
    dataset = build_dataset(cfg.data.test)

    write_info(dataset, bin_path, width, height, info_name)
    write_meta_info(dataset, meta_path, info_meta_name)

    print("Get info done!")


if __name__ == '__main__':
    main()