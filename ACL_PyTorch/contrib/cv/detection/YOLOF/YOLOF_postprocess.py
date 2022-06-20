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
import os

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
    parser.add_argument('--bin_data_path', default="result/")
    parser.add_argument('--meta_info_path', default="yolof_meta.info")
    parser.add_argument('--num_classes', default=81, type=int)

    args = parser.parse_args()

    opts = ['MODEL.WEIGHTS', args.pth_path, "MODEL.DEVICE", "cpu"]
    config.merge_from_list(opts)

    cls = runner_decrator(RUNNERS.get(config.TRAINER.NAME))
    evaluator = cls.build_evaluator(config, dataset_name, cls.build_test_loader(config).dataset)
    evaluator.reset()
    bin_data_path = args.bin_data_path + os.listdir(args.bin_data_path)[0] + "/"

    with open(args.meta_info_path, "r") as fp:
        for line in fp:
            values = line.split()
            file_name = values[0]
            batch_size = (len(values) - 1) // 3
            nmsed_boxes_batch = np.fromfile("{0}{1}_output_{2}.bin".format(bin_data_path, file_name, 0),
                                            dtype=np.float32).reshape(batch_size, -1, 4)
            nmsed_scores_batch = np.fromfile("{0}{1}_output_{2}.bin".format(bin_data_path, file_name, 1),
                                             dtype=np.float32).reshape(batch_size, -1)
            nmsed_classes_batch = np.fromfile("{0}{1}_output_{2}.bin".format(bin_data_path, file_name, 2),
                                              dtype=np.int64).reshape(batch_size, -1)
            last_image = ""
            for i in range(batch_size):
                img_name = values[i * 3 + 1]
                if img_name == last_image:
                    break
                last_image = img_name
                last_img_name = img_name
                height = int(values[i * 3 + 2])
                width = int(values[i * 3 + 3])
                nmsed_boxes = nmsed_boxes_batch[i]
                nmsed_scores = nmsed_scores_batch[i]
                nmsed_classes = nmsed_classes_batch[i]
                result = Instances(const_shape)
                result.pred_boxes = Boxes(torch.tensor(nmsed_boxes))
                result.scores = torch.tensor(nmsed_scores)
                result.pred_classes = torch.tensor(nmsed_classes)
                r = detector_postprocess(result, height, width)
                r = {"instances": r}
                _input = {"image_id": int(img_name)}
                evaluator.process([_input], [r])
    print(evaluator.evaluate())
    print(evaluator._dump_infos[0]['summary'])
