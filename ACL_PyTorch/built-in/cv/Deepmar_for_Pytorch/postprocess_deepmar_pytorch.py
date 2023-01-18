# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import argparse
import json
import numpy as np


def attribute_evaluate_lidw(gt_result, pt_result):
    """
    Input:
    gt_result, pt_result, N*L, with 0/1
    Output:
    result
    a dictionary, including label-based and instance-based evaluation
    label-based: label_pos_acc, label_neg_acc, label_acc
    instance-based: instance_acc, instance_precision, instance_recall, instance_F1
    """
    # obtain the label-based and instance-based accuracy
    # compute the label-based accuracy
    if gt_result.shape != pt_result.shape:
        print('Shape beteen groundtruth and predicted results are different')
    # compute the label-based accuracy
    result = {}
    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == 0).astype(float), axis=0)
    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == 0).astype(float) * (pt_result == 0).astype(float), axis=0)
    label_pos_acc = 1.0*pt_pos/gt_pos
    label_neg_acc = 1.0*pt_neg/gt_neg
    label_acc = (label_pos_acc + label_neg_acc)/2
    result['label_pos_acc'] = label_pos_acc
    result['label_neg_acc'] = label_neg_acc
    result['label_acc'] = label_acc
    # compute the instance-based accuracy
    # precision
    gt_pos = np.sum((gt_result == 1).astype(float), axis=1)
    pt_pos = np.sum((pt_result == 1).astype(float), axis=1)
    floatersect_pos = np.sum((gt_result == 1).astype(float)*(pt_result == 1).astype(float), axis=1)
    union_pos = np.sum(((gt_result == 1)+(pt_result == 1)).astype(float),axis=1)
    # avoid empty label in predicted results
    cnt_eff = float(gt_result.shape[0])
    for iter, key in enumerate(gt_pos):
        if key == 0:
            union_pos[iter] = 1
            pt_pos[iter] = 1
            gt_pos[iter] = 1
            cnt_eff = cnt_eff - 1
            continue
        if pt_pos[iter] == 0:
            pt_pos[iter] = 1
    instance_acc = np.sum(floatersect_pos/union_pos)/cnt_eff
    instance_precision = np.sum(floatersect_pos/pt_pos)/cnt_eff
    instance_recall = np.sum(floatersect_pos/gt_pos)/cnt_eff
    floatance_F1 = 2*instance_precision*instance_recall/(instance_precision+instance_recall)
    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = floatance_F1
    return result


def postprocess_deepmar(npu_result, label_file):
    with open(label_file, 'r') as json_file:
        image_label = json.load(json_file)
    bin_files = os.listdir(npu_result)
    counts = len(bin_files)
    gt_result = np.zeros((counts, 35))
    pt_result = np.zeros((counts, 35))

    for index, bin_file in enumerate(bin_files):
        result = np.fromfile(os.path.join(npu_result, bin_file), dtype='float32').reshape((35, ))
        result[result >= 0] = 1
        result[result < 0] = 0
        pt_result[index, :] = result

        # label
        key = bin_file[:bin_file.rfind('_')] + '.png'
        label = image_label[key]
        label = np.array(label, dtype=np.uint8)
        gt_result[index, :] = label

    result = attribute_evaluate_lidw(gt_result, pt_result)
    print('-'*80)
    print('instance_acc: {}'.format(result['instance_acc']))
    print('instance_precision: {}'.format(result['instance_precision']))
    print('instance_recall: {}'.format(result['instance_recall']))
    print('instance_F1: {}'.format(result['instance_F1']))
    print('-' * 80)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--npu_result', type=str, default="./mae_finetuned_vit_base.pth")
    parser.add_argument('--label_file', type=str, default="./mae_dynamicbs.onnx")
    args = parser.parse_args()

    npu_result = os.path.abspath(args.npu_result)
    label_file = os.path.abspath(args.label_file)
    postprocess_deepmar(npu_result, label_file)

