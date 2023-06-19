# coding = utf-8
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
import json
import time
import numpy as np

np.set_printoptions(threshold=sys.maxsize)



def cre_groundtruth_dict_fromtxt(val_label_path):
    """
    读取标签文件信息
    :输入：标签文件地址
    :输出: dict结构，key：图片名称，value：图片分类
    """
    img_label_dict_inp = {}
    with open(val_label_path, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            imgName = temp[0].split(".")[0]
            imgLab = temp[1]
            img_label_dict_inp[imgName] = imgLab
    return img_label_dict_inp


def load_statistical_predict_result(filepath):
    """
    function:
    the prediction result file data extraction
    input:
    result file:filepath
    output:
    n_label:numble of label
    data_vec: the probabilitie of prediction in the 1000
    :return: probabilities, numble of label, in_type, color
    """
    with open(filepath, 'r')as f:
        data = f.readline()
        temp = data.strip().split(" ")
        n_label = len(temp)
        data_vec = np.zeros((n_label), dtype=np.float32)
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
                                            result_path, json_name,
                                            img_label_dict_inp, topn=5):
    """
    :param prediction_file_path: 推理结果路径
    :param result_path: 后处理结果保存的json文件路径
    :param json_name: 结果文件的名字
    :param img_label_dict_inp: 真实标签结果，dict形式，key为图片名称，value是标签
    :param topn: 1~5
    :return: NA
    """
    writer = open(os.path.join(result_path, json_name), 'w')
    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []

    count = 0
    res_count = 0
    n_labels = ""
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
        gt = img_label_dict_inp[img_name]
        if n_labels == 1000:
            realLabel = int(gt)
        elif n_labels == 1001:
            realLabel = int(gt) + 1
        else:
            realLabel = int(gt)


        res_count = min(len(sort_index), topn)
        for i in range(res_count):
            if (str(realLabel) == str(sort_index[i])):
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
        for i in range(res_count):
            table_dict["value"].append({"key": "Top" + str(i + 1) + " accuracy",
                                        "value": str(round(accuracy[i] * 100, 2)) + '%'})

        json.dump(table_dict, writer)
    writer.close()


if __name__ == '__main__':
    start = time.time()
    try:
        # infer result file path
        infer_result_path = sys.argv[1] 

        # annotation files path, "val_label.txt"
        annotation_file_path = sys.argv[2]

        # the path to store the results json path
        result_json_path = sys.argv[3]

        # result json file name
        json_file_name = sys.argv[4]
    except IndexError:
        print("Stopped!")
        exit(1)

    if not (os.path.exists(infer_result_path)):
        print("infer result path does not exist.")
        os.makedirs(infer_result_path)

    if not (os.path.exists(annotation_file_path)):
        print("Ground truth file does not exist.")
        os.makedirs(annotation_file_path)

    if not (os.path.exists(result_json_path)):
        print("Result folder doesn't exist.")
        os.makedirs(result_json_path)

    img_label_dict = cre_groundtruth_dict_fromtxt(annotation_file_path)
    create_visualization_statistical_result(infer_result_path,
                                            result_json_path, json_file_name,
                                            img_label_dict, topn=5)

    elapsed = (time.time() - start)
