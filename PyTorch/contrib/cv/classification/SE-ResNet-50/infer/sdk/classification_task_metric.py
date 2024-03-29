# coding=utf-8
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


import json
import os
import sys

import numpy as np

np.set_printoptions(threshold=sys.maxsize)

LABEL_FILE = "HiAI_label.json"


def cre_groundtruth_dict(gtfile_path):
    """
    :param filename: file contains the imagename and label number
    :return: dictionary key imagename, value is label number
    """
    img_gt_dict = {}
    for gtfile in os.listdir(gtfile_path):
        if gtfile != LABEL_FILE:
            with open(os.path.join(gtfile_path, gtfile), 'r') as f:
                gt = json.load(f)
                image_name = os.path.splitext(gtfile.split('/')[-1])
                img_gt_dict[image_name] = gt["image"]["annotations"][0]["category_id"]
    return img_gt_dict


def cre_groundtruth_dict_fromtxt(gtfile_path):
    """
    :param filename: file contains the imagename and label number
    :return: dictionary key imagename, value is label number
    """
    img_gt_dict = {}
    with open(gtfile_path, 'r')as f:
        for line in f.readlines():
            image_line_info = line.strip().split(" ")
            img_name = image_line_info[0].split(".")[0]
            img_gt_dict[img_name] = image_line_info[1]
    return img_gt_dict
   

def load_statistical_predict_result(filepath):
    """
    function:
    the prediction esult file data extraction
    input:
    result file:filepath
    output:
    n_label:numble of label
    data_vec: the probabilitie of prediction in the 1000
    :return: probabilities, numble of label
    """
    with open(filepath, 'r')as f:
        label_info = f.readline().strip().split(" ")
        data_vec = np.zeros((len(label_info)), dtype=np.float32)
        n_label = len(label_info)
        if n_label == 0:
            in_type = f.readline()
            color = f.readline()
        else:
            for ind, cls_ind in enumerate(label_info):
                data_vec[ind] = np.int32(cls_ind)
    return data_vec, n_label


def create_visualization_statistical_result(prediction_file_path,
                                            result_store_path, json_file_name,
                                            img_gt_dict, topn=5):
    """
    :param prediction_file_path:
    :param result_store_path:
    :param json_file_name:
    :param img_gt_dict:
    :param topn:
    :return:
    """
    writer = open(os.path.join(result_store_path, json_file_name), 'w')
    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []

    count = 0
    res_cnt = 0
    n_labels = ""
    count_hit = np.zeros(topn)
    for tfile_name in os.listdir(prediction_file_path):
        count += 1
        temp = tfile_name.split('.')[0]
        index = temp.rfind('_')
        img_name = temp[:index]
        filepath = os.path.join(prediction_file_path, tfile_name)
        prediction, n_labels = load_statistical_predict_result(filepath)

        if n_labels == 1001:
            real_label = int(img_gt_dict[img_name]) + 1
        else:
            real_label = int(img_gt_dict[img_name])

        res_cnt = min(len(prediction), topn)
        for i in range(res_cnt):
            if real_label == int(prediction[i]):
                count_hit[i] += 1
                break
    if 'value' not in table_dict.keys():
        print("the item value does not exist!")
    else:
        table_dict["value"].extend(
            [{"key": "Number of images", "value": str(count)},
             {"key": "Number of classes", "value": str(n_labels)}])
        accuracy = np.cumsum(count_hit) / count if count else 0
        for i in range(res_cnt):
            table_dict["value"].append({"key": "Top" + str(i + 1) + " accuracy",
                                        "value": str(
                                            round(accuracy[i] * 100, 2)) + '%'})
        json.dump(table_dict, writer)
    writer.close()


if __name__ == '__main__':
    if len(sys.argv) == 5:
        # txt file path
        folder_davinci_target = sys.argv[1]       
        # annotation files path, "val_label.txt"
        annotation_file_path = sys.argv[2]                
        # the path to store the results json path
        result_json_path = sys.argv[3]
        # result json file name
        json_file_name = sys.argv[4]
    else:
        print("Please enter target file result folder | ground truth label file | result json file folder | "
              "result json file name, such as ./result val_label.txt . result.json")
        exit(1)

    if not os.path.exists(folder_davinci_target):
        print("Target file folder does not exist.")

    if not os.path.exists(annotation_file_path):
        print("Ground truth file does not exist.")

    if not os.path.exists(result_json_path):
        print("Result folder doesn't exist.")

    img_label_dict = cre_groundtruth_dict_fromtxt(annotation_file_path)
    create_visualization_statistical_result(folder_davinci_target,
                                            result_json_path, json_file_name,
                                            img_label_dict, topn=5)

