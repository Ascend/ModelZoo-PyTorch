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
import mmcv
from mmdet.datasets import build_dataset
import pickle as pk

ann_file = '/annotations/instances_val2017.json'
img_prefix = '/val2017/'

if __name__ == '__main__':
    image_src_path = sys.argv[1]
    config_path = sys.argv[2]
    bin_path = sys.argv[3]
    meta_path = sys.argv[4]
    info_name = sys.argv[5]
    info_meta_name = sys.argv[6]
    width = int(sys.argv[7])
    height = int(sys.argv[8])

    cfg = mmcv.Config.fromfile(config_path)
    cfg.data.test.ann_file = image_src_path + ann_file
    cfg.data.test.img_prefix = image_src_path + img_prefix
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)

    fp1 = open(info_name, "w")
    fp2 = open(info_meta_name, "w")

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
    fp1.close()
    fp2.close()
    print("Get info done!")
