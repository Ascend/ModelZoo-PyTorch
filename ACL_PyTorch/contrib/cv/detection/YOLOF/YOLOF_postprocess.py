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
import numpy as np
import argparse

import torch
from cvpods.structures import Boxes, Instances
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.engine import RUNNERS
from cvpods.evaluation import build_evaluator

import sys

sys.path.append("{0}/YOLOF/playground/detection/coco/yolof/yolof.cspdarknet53.DC5.9x/".format(sys.path[0]))
from config import config

const_shape = (608, 608)
dataset_name = "coco_2017_val"


def runner_decrator(cls):
    def custom_build_evaluator(cls, cfg, dataset_name, dataset):
        return build_evaluator(cfg, dataset_name, dataset, None, dump=True)

    cls.build_evaluator = classmethod(custom_build_evaluator)
    return cls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', default="YOLOF_CSP_D_53_DC5_9x.pth")
    parser.add_argument('--bin_data_path', default="result/dumpOutput_device0/")
    parser.add_argument('--meta_info_path', default="yolof_meta.info")
    parser.add_argument('--num_classes', default=81, type=int)

    args = parser.parse_args()

    opts = ['MODEL.WEIGHTS', args.pth_path, "MODEL.DEVICE", "cpu"]
    config.merge_from_list(opts)

    results = []
    cls = runner_decrator(RUNNERS.get(config.TRAINER.NAME))
    evaluator = cls.build_evaluator(config, dataset_name, cls.build_test_loader(config).dataset)
    evaluator.reset()
    with open(args.meta_info_path, "r") as fp:
        for line in fp:
            _, file_name, height, width = line.split()
            height = int(height)
            width = int(width)
            nmsed_boxes = np.fromfile("{0}{1}_{2}.bin".format(args.bin_data_path, file_name, 1),
                                      dtype=np.float32).reshape(-1, 4)
            nmsed_scores = np.fromfile("{0}{1}_{2}.bin".format(args.bin_data_path, file_name, 2), dtype=np.float32)
            nmsed_classes = np.fromfile("{0}{1}_{2}.bin".format(args.bin_data_path, file_name, 3), dtype=np.int64)
            result = Instances(const_shape)
            result.pred_boxes = Boxes(torch.tensor(nmsed_boxes))
            result.scores = torch.tensor(nmsed_scores)
            result.pred_classes = torch.tensor(nmsed_classes)
            r = detector_postprocess(result, height, width)
            r = {"instances": r}
            _input = {"image_id": int(file_name)}
            evaluator.process([_input], [r])

    evaluator.evaluate()
    print(evaluator._dump_infos[0]['summary'])
