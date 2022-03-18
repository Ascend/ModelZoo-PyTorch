# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import argparse
import tqdm
import yaml
import sys

import numpy as np

from PIL import Image
from tabulate import tabulate
from easydict import EasyDict as ed

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.eval_functions import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/PraNet.yaml')
    return parser.parse_args()

def eval(opt):
    if os.path.isdir(opt.Eval.result_path) is False:
        os.makedirs(opt.Eval.result_path)

    # method = PraNet
    method = os.path.split(opt.Eval.pred_path)[-1]
    Thresholds = np.linspace(1, 0, 256)
    headers = ['meanDic', 'meanIoU']
    results = []

    print('#' * 20, 'Start Evaluation', '#' * 20)
    for dataset in opt.Eval.datasets:
        pred_path = os.path.join(opt.Eval.pred_path, dataset)
        gt_path = os.path.join(opt.Eval.gt_path, dataset, 'masks')

        preds = os.listdir(pred_path)
        gts = os.listdir(gt_path)

        preds.sort()
        gts.sort()

        threshold_IoU = np.zeros((len(preds), len(Thresholds)))
        threshold_Dice = np.zeros((len(preds), len(Thresholds)))

        Smeasure = np.zeros(len(preds))
        wFmeasure = np.zeros(len(preds))
        MAE = np.zeros(len(preds))

        for i, sample in enumerate(zip(preds, gts)):
            pred, gt = sample
            assert os.path.splitext(pred)[0] == os.path.splitext(gt)[0]

            pred_mask = np.array(Image.open(os.path.join(pred_path, pred)))
            gt_mask = np.array(Image.open(os.path.join(gt_path, gt)))

            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]
            
            assert pred_mask.shape == gt_mask.shape

            gt_mask = gt_mask.astype(np.float64) / 255
            gt_mask = (gt_mask > 0.5).astype(np.float64)

            pred_mask = pred_mask.astype(np.float64) / 255

            Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
            wFmeasure[i] = original_WFb(pred_mask, gt_mask)
            MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

            threshold_E = np.zeros(len(Thresholds))
            threshold_F = np.zeros(len(Thresholds))
            threshold_Pr = np.zeros(len(Thresholds))
            threshold_Rec = np.zeros(len(Thresholds))
            threshold_Iou = np.zeros(len(Thresholds))
            threshold_Spe = np.zeros(len(Thresholds))
            threshold_Dic = np.zeros(len(Thresholds))

            for j, threshold in enumerate(Thresholds):
                threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[j] = Fmeasure_calu(pred_mask, gt_mask, threshold)

                Bi_pred = np.zeros_like(pred_mask)
                Bi_pred[pred_mask >= threshold] = 1
                threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)
            
            threshold_Dice[i, :] = threshold_Dic
            threshold_IoU[i, :] = threshold_Iou

        result = []

        column_Dic = np.mean(threshold_Dice, axis=0)
        meanDic = np.mean(column_Dic)
        maxDic = np.max(column_Dic)

        column_IoU = np.mean(threshold_IoU, axis=0)
        meanIoU = np.mean(column_IoU)
        maxIoU = np.max(column_IoU)

        result.extend([meanDic, meanIoU])
        results.append([dataset, *result])

        csv = os.path.join(opt.Eval.result_path, 'result_' + dataset + '.csv')
        if os.path.isfile(csv) is True:
            csv = open(csv, 'a')
        else:
            csv = open(csv, 'w')
            csv.write(', '.join(['method', *headers]) + '\n')

        out_str = method + ','
        for metric in result:
            out_str += '{:.4f}'.format(metric) + ','
        out_str += '\n'

        csv.write(out_str)
        csv.close()
    print(tabulate(results, headers=['dataset', *headers], floatfmt=".3f"))
    print("#"*20, "End Evaluation", "#"*20)

if __name__ == "__main__":
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))
    eval(opt)

