# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import operator
import os
import sys
import argparse
import json
from tqdm import tqdm
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2

from yolox.data import COCODataset, ValTransform
from yolox.evaluators import COCOEvaluator
from yolox.utils.boxes import postprocess
from yolox.utils.demo_utils import demo_postprocess

sys.path.append('./YOLOX')


def get_output_data(dump_dir, idx, dtype=np.float32):
    input_file_2 = os.path.join(dump_dir, "{:0>12d}_0.bin".format(idx))
    input_file_1 = os.path.join(dump_dir, "{:0>12d}_1.bin".format(idx))
    input_data_1 = np.fromfile(input_file_1, dtype=dtype).reshape([-1, 8400, 6])
    # reshape中的400需要与add_nms_op.py脚本中的max_output_boxes_per_class一致
    input_data_2 = np.fromfile(input_file_2, dtype=np.int64).reshape([-1, 400, 3])

    return input_data_1, input_data_2


def main():
    parser = argparse.ArgumentParser(description='YOLOX Postprocess')
    parser.add_argument('--dataroot', dest='dataroot',
                        help='data root dirname', default='/opt/npu/coco',
                        type=str)
    parser.add_argument('--dump_dir', dest='dump_dir',
                        help='dump dir for bin files', default='./result/',
                        type=str)
    parser.add_argument('--batch', dest='batch', help='batch for dataloader', default=1, type=int)
    opt = parser.parse_args()

    valdataset = COCODataset(
        data_dir=opt.dataroot,
        json_file='instances_val2017.json',
        name="val2017",
        img_size=(640, 640),
        preproc=ValTransform(legacy=False),
    )
    sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {"num_workers": 8, "pin_memory": True, "sampler": sampler, "batch_size": opt.batch}

    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    data_list = []
    coco_evaluator = COCOEvaluator(val_loader, img_size=(640, 640), confthre=0.001, nmsthre=0.65, num_classes=80)
    statistics = torch.FloatTensor([10, 10, max(len(val_loader) - 1, 1)])

    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(tqdm(val_loader)):
        nms_select_output = []
        # 现模型改图后有俩个输出节点，一个为原输出节点，一个为nms输出节点，分别读取两个节点的推理结果
        outputs1, outputs2 = get_output_data(opt.dump_dir, cur_iter)

        # 由于om的nmsv7算子中输出不足要求的最大输出值会用-1进行填充，故需要筛选indices
        for i in outputs2[:, :, 2][0]:
            if i != -1 and i not in nms_select_output:
                nms_select_output.append(i)

        # 将筛选出的合规indices，用于在模型推理结果中gather出符合要求的prediction
        outputs = outputs1[:, nms_select_output, :]
        final_output = torch.from_numpy(outputs)

        # 将推理结果的格式转换为pycocotools计算精度要求的list套dict的格式
        output_list2coco_format = coco_evaluator.convert_to_coco_format(final_output, info_imgs, ids)
        data_list.extend(output_list2coco_format)
    results = coco_evaluator.evaluate_prediction(data_list, statistics)
    print(results)


if __name__ == "__main__":
    main()
