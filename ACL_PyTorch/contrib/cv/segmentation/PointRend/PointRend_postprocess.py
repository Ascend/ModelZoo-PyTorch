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

import os
import sys
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval
from detectron2.evaluation.cityscapes_evaluation import CityscapesSemSegEvaluator


def cvt_gt_path(path):
    tmp = path.split('.')[0].split('/')
    tmp[-4] = 'gtFine'
    tmp_name = tmp[-1].split('_')[: -1]
    tmp_name.append('gtFine_labelIds')
    tmp_name = '_'.join(tmp_name)
    tmp[-1] = tmp_name
    return '/'.join(tmp) + '.png'


def cvt_pred_name(path):
    name = path.split('/')[-1]
    name = name.split('.')
    name[0] = name[0] + '_pred'
    return '.'.join(name)


def sem_seg_postprocess(result, h, w):
    result = result.expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(h, w), mode="bilinear", align_corners=False
    )[0]
    return result


def eval_acc(src_path, pred_path):
    input_dir = src_path + '/leftImg8bit/val' 
    target_dir = src_path + '/gtFine/val'
    citys = os.listdir(target_dir)
    citys.sort()
    input_paths = []
    target_paths = []
    for city in citys:
        city_path = os.path.join(input_dir, city)
        files = os.listdir(city_path)
        files.sort()
        for file in files:
            sem_seg_path = cvt_gt_path(os.path.join(city_path, file))
            tmp_file = {'file_name' : os.path.join(city_path, file)}
            input_paths.append(tmp_file)
            target_paths.append(sem_seg_path)

    evaluator = CityscapesSemSegEvaluator("cityscapes_fine_sem_seg_val")
    evaluator.reset()

    bin_data_path = pred_path
    files = os.listdir(bin_data_path)
    files.sort()
    idx = 0
    for file in files:
        file_path = os.path.join(bin_data_path, file)
        outputs = np.fromfile(file_path, dtype=np.float32).reshape(-1, 19, 1024, 2048)
        tmp_file = []
        results = []
        for i in range(outputs.shape[0]):
            r = sem_seg_postprocess(torch.as_tensor(outputs[i]), 1024, 2048)
            results.append({"sem_seg": r})
            tmp_file.append(input_paths[idx])
            idx += 1
            print('postprocessing image {}'.format(idx), end='\r')
        evaluator.process(tmp_file, results)

    pred_paths = [os.path.join(evaluator._temp_dir, cvt_pred_name(x['file_name'])) for x in input_paths]

    cityscapes_eval.args.predictionPath = os.path.abspath(evaluator._temp_dir)
    cityscapes_eval.args.predictionWalk = None
    cityscapes_eval.args.JSONOutput = False
    cityscapes_eval.args.colorized = False
    metrics = cityscapes_eval.evaluateImgLists(
        pred_paths, target_paths, cityscapes_eval.args
    )
    ret = OrderedDict()
    ret["sem_seg"] = {
        "IoU": 100.0 * metrics["averageScoreClasses"],
        "iIoU": 100.0 * metrics["averageScoreInstClasses"],
        "IoU_sup": 100.0 * metrics["averageScoreCategories"],
        "iIoU_sup": 100.0 * metrics["averageScoreInstCategories"],
    }
    evaluator._working_dir.cleanup()
    print('acc result = {}'.format(ret["sem_seg"]["IoU"]))

if __name__ == '__main__':
    src_path = sys.argv[1]
    pred_path = sys.argv[2]
    eval_acc(src_path, pred_path)