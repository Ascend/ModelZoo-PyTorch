"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import torch
import torch.nn.functional as F
import os
import sys


def cre_groundtruth_dict_fromtxt(gtfile_path):
    """
    :param filename: file contains the imagename and label number
    :return: dictionary key imagename, value is label number
    """
    img_gt_dict = {}
    with open(gtfile_path, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split("/")[-1]
            temp = temp.split()
            imgName = temp[0].split(".")[0]
            imgLab = temp[1]
            img_gt_dict[imgName] = imgLab
    return img_gt_dict


def cre_groundtruth_list_fromtxt(gtfile_path):
    """
    :param filename: file contains the imagename and label number
    :return: dictionary key imagename, value is label number
    """
    gt_list = []
    with open(gtfile_path, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split()[-1]
            gt_list.append(int(temp))
            
    return gt_list


def calc_acc_bs1(dirname, img_ge_dict):
    output_folder = os.listdir(dirname)
    output_files = os.listdir(os.path.join(dirname, output_folder[0]))
    ground_truth = []
    labels = []
    for f in output_files:
        real_name = f.split('_')[0]
        if real_name in img_ge_dict:
            ground_truth.append(img_ge_dict[real_name])
        file_name = os.path.join(dirname, output_folder[0], f)
        file = open(file_name, 'r')
        content = file.read()
        file.close()
        res = list(map(float, content.split()))
        res_arr = np.array(res)
        pred_label = np.argmax(res_arr)
        labels.append(pred_label)

    total = len(ground_truth)
    correct = 0.
    for i in range(total):
        if int(ground_truth[i]) == int(labels[i]):
            correct += 1
    print('om model accuracy:', correct / total)


def calc_acc_bs16(dirname, gt_list):
    output_folder = os.listdir(dirname)
    output_files = os.listdir(os.path.join(dirname, output_folder[0]))
    file_count = len(output_files)
    labels = []
    for i in range(file_count):
        if i < 10:
            file_name = "bin_batch00" + str(i) + "_output_0.txt"
        elif i >= 10 and i < 100:
            file_name = "bin_batch0" + str(i) + "_output_0.txt"
        else:
            file_name = "bin_batch" + str(i) + "_output_0.txt"
        f = open(os.path.join(dirname, output_folder[0], file_name))
        for line in f.readlines():
            content_list = list(map(float, line.split()))
            arr = np.array(content_list)
            labels.append(np.argmax(arr))

    total = len(labels)
    correct = 0.
    for i in range(total):
        if gt_list[i] == labels[i]:
            correct += 1
    print('om model accuracy:', correct / total)

if __name__ == '__main__':
    name2label = sys.argv[1]
    infer_res = sys.argv[2]
    batch_size = sys.argv[3]
    if batch_size.endswith("16"):
        img_gt_list = cre_groundtruth_list_fromtxt(name2label)
        calc_acc_bs16(infer_res, img_gt_list)
    else:
        img_gt_dict = cre_groundtruth_dict_fromtxt(name2label)
        calc_acc_bs1(infer_res, img_gt_dict)
