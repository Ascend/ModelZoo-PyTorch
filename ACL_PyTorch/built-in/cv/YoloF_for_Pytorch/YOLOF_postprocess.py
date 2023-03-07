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

import mmcv
import numpy as np
import argparse
from mmdet.core import bbox2result
from mmdet.datasets import build_dataset
from tqdm import tqdm
ann_file = '/annotations/instances_val2017.json'
img_prefix = '/val2017/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/opt/npu/coco")
    parser.add_argument('--model_config', default="mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py")
    parser.add_argument('--bin_data_path', default="result/")
    parser.add_argument('--meta_info_path', default="yolof_meta.info")
    parser.add_argument('--num_classes', default=81)

    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.model_config)
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = args.dataset_path + ann_file
    cfg.data.test.img_prefix = args.dataset_path + img_prefix
    dataset = build_dataset(cfg.data.test)

    num_classes = int(args.num_classes)
    outputs = []
    with open(args.meta_info_path, "r") as fp:
        for line in tqdm(fp):
            _, file_path, scalar = line.split()
            scalar = float(scalar)
            file_name = file_path.split("/")[1].replace(".bin", "")
            result_list = [
                np.fromfile("{0}{1}_{2}.bin".format(args.bin_data_path, file_name, 0), dtype=np.float32).reshape(-1, 5),
                np.fromfile("{0}{1}_{2}.bin".format(args.bin_data_path, file_name, 1), dtype=np.int32)]
            result_list[0][..., :4] /= scalar
            bbox_result = bbox2result(result_list[0], result_list[1], num_classes)
            outputs.append(bbox_result)
    eval_kwargs = {'metric': ['bbox']}
    res = dataset.evaluate(outputs, **eval_kwargs)
    resstr = "acc: {}\n".format(res['bbox_mAP'])
    with open("results.txt", "a") as resfile:
        resfile.write(resstr)
