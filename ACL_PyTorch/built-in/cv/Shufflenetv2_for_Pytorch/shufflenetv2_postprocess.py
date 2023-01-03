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

import os
import sys
import json
import time

import numpy as np

def cre_groundtruth_dict_fromtxt(gtfile_path):
    """
    :param filename: file contains the imagename and label number
    :return: dictionary key imagename, value is label number
    """
    img_gt_dict = {}
    with open(gtfile_path, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            imgName = temp[0].split(".")[0]
            imgLab = temp[1]
            img_gt_dict[imgName] = imgLab
    return img_gt_dict


def load_statistical_predict_result(filepath):
    """
    :param filepath: filepath to prediction file
    :return: probabilities, number of label
    """
    with open(filepath, 'r')as f:
        data = f.readline()
        temp = data.strip().split(" ")
        n_label = len(temp)
        if data == '':
            n_label = 0
        data_vec = np.zeros((n_label), dtype=np.float32)
        if n_label == 0:
            _ = f.readline()
            _ = f.readline()
        else:
            for ind, prob in enumerate(temp):
                data_vec[ind] = np.float32(prob)
    return data_vec, n_label


def create_visualization_statistical_result(prediction_file_path,
                                            json_file,
                                            img_gt_dict, topn=5):
    """
    :param prediction_file_path:
    :param json_file:
    :param img_gt_dict:
    :param topn:
    :return:
    """
    writer = open(json_file, 'w')
    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []

    count = 0
    resCnt = 0
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
            realLabel = int(gt)
        elif n_labels == 1001:
            realLabel = int(gt) + 1
        else:
            realLabel = int(gt)
        resCnt = min(len(sort_index), topn)
        for i in range(resCnt):
            if str(realLabel) == str(sort_index[i]):
                count_hit[i] += 1
                break
    if 'value' not in table_dict.keys():
        print("the item value does not exist!")
    else:
        table_dict["value"].extend(
            [{"key": "Number of images", "value": str(count)},
             {"key": "Number of classes", "value": str(n_labels)}])
        if count == 0:
            accuracy = 0
        else:
            accuracy = np.cumsum(count_hit) / count
        for i in range(resCnt):
            table_dict["value"].append({"key": "Top" + str(i + 1) + " accuracy",
                                        "value": str(
                                            round(accuracy[i] * 100, 2)) + '%'})
        json.dump(table_dict, writer)
    writer.close()


if __name__ == '__main__':
    start = time.time()

    inference_result_dir = sys.argv[1] # path to output bin files       
    annotation_file_path = sys.argv[2] # annotation files path, "val_label.txt"                
    json_file_name = sys.argv[3] # metric result json file name

    if not os.path.exists(inference_result_dir):
        print("Inference result dir does not exist.")
    if not os.path.exists(annotation_file_path):
        print("Ground truth file does not exist.")

    img_label_dict = cre_groundtruth_dict_fromtxt(annotation_file_path)
    create_visualization_statistical_result(inference_result_dir,
                                            json_file_name,
                                            img_label_dict, topn=5)

    elapsed = (time.time() - start)
    print("Time used:", elapsed)
    