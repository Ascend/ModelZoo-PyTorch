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

import argparse
import json
import os
import sys
import stat
import numpy as np


np.set_printoptions(threshold=sys.maxsize)

LABEL_FILE = "HiAI_label.json"


def gen_file_name(img_name):
    full_name = img_name.split('/')[-1]
    index = full_name.rfind('.')
    return full_name[:index]


def cre_groundtruth_dict(filename):
    """
    :param filename: file contains the image_name and label number
    :return: dictionary key image_name, value is label number
    """
    img_gt_dict = {}
    for gt_file in os.listdir(filename):
        if gt_file != LABEL_FILE:
            with open(os.path.join(filename, gt_file), 'r') as f:
                gt = json.load(f)
                ret = gt["image"]["annotations"][0]["category_id"]
                img_gt_dict[gen_file_name(gt_file)] = ret
    return img_gt_dict


def cre_groundtruth_dict_fromtxt(filename):
    """
    :param filename: file contains the image_name and label number
    :return: dictionary key image_name, value is label number
    """
    img_gt_dict = {}
    with open(filename, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            imgName = temp[0].split(".")[0]
            imgLab = temp[1]
            img_gt_dict[imgName] = imgLab
    return img_gt_dict


def load_statistical_predict_result(filepath):
    """
    function:
    the prediction result file data extraction
    input:
    result file:filepath
    output:
    n_label:number of label
    data_vec: the probabilities of prediction in the 1000
    :return: probabilities, number of label, in_type, color
    """
    with open(filepath, 'r')as f:
        data = f.readline()
        temp = data.strip().split(" ")
        n_label = len(temp)
        if data == '':
            n_label = 0
        data_vec = np.zeros(n_label, dtype=np.float32)
        in_type = ''
        color = ''
        if n_label == 0:
            in_type = f.readline()
            color = f.readline()
        else:
            for ind, prob in enumerate(temp):
                data_vec[ind] = np.float32(prob)
    return data_vec, n_label, in_type, color


def create_visualization_statistical_result(prediction_file_path,
                                            result_file, img_gt_dict,
                                            topn=5):
    """
    :param prediction_file_path:
    :param result_file:
    :param img_gt_dict:
    :param topn:
    :return:
    """
    writer = os.open(result_file, os.O_CREAT | os.O_WRONLY, stat.S_IWUSR)
    table_dict = {"title": "Overall statistical evaluation", "value": []}

    count = 0
    res_cnt = 0
    n_labels = None
    count_hit = np.zeros(topn)
    for tfile_name in os.listdir(prediction_file_path):
        count += 1
        temp = tfile_name.split('.')[0]
        index = temp.rfind('_')
        img_name = temp[:index]
        filepath = os.path.join(prediction_file_path, tfile_name)
        ret = load_statistical_predict_result(filepath)
        prediction = ret[0]
        n_labels = ret[1]
        sort_index = np.argsort(-prediction)
        gt = img_gt_dict[img_name]
        if n_labels == 1000:
            real_label = int(gt)
        elif n_labels == 1001:
            real_label = int(gt) + 1
        else:
            real_label = int(gt)

        res_cnt = min(len(sort_index), topn)
        for i in range(res_cnt):
            if str(real_label) == str(sort_index[i]):
                count_hit[i] += 1
                break

    if 'value' not in table_dict.keys():
        print("the item value does not exist!")
    else:
        table_dict["value"].extend(
            [{"key": "Number of images", "value": str(count)},
             {"key": "Number of classes", "value": str(n_labels)}])
        accuracy = np.cumsum(count_hit) / count
        for i in range(res_cnt):

            os.write(writer, str({"key": "Top" + str(i + 1) + " accuracy", "value": str(
                                            round(accuracy[i] * 100, 2)) + '%'}).encode())

    os.close(writer)


def check_args(arg):
    if not (os.path.exists(arg.anno_file)):
        print("annotation file:{} does not exist.".format(arg.anno_file))
        exit()
    if not (os.path.exists(arg.benchmark_out)):
        print("benchmark output:{} does not exist.".format(arg.benchmark_out))
        exit()
    return arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precision statistics of Vision model')
    parser.add_argument("--anno_file", default="./HiAI_label.json", help='annotation file')
    parser.add_argument("--benchmark_out", default="result/dumpOutput_device0", help='Benchmark output directory')
    parser.add_argument("--result_file", default="./result.txt", help='Output json file')
    args = parser.parse_args()
    args = check_args(args)
    if args.anno_file.endswith('txt'):
        img_label_dict = cre_groundtruth_dict_fromtxt(args.anno_file)
    else:
        img_label_dict = cre_groundtruth_dict(args.anno_file)
    create_visualization_statistical_result(args.benchmark_out, args.result_file, img_label_dict, topn=5)
