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

import os
import stat
import sys
import json
import time
import argparse
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

def cre_groundtruth_dict_fromtxt(val_label_path):
    """
    读取标签文件信息
    :输入：标签文件地址
    :输出: dict结构，key：图片名称，value：图片分类
    """
    img_label_dict = {}
    with open(val_label_path, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            imgName = temp[0].split(".")[0]
            imgLab = temp[1]
            img_label_dict[imgName] = imgLab
    return img_label_dict


def load_statistical_predict_result(filepath, tfile_type):
    """
    function:
    the prediction result file data extraction
    input:
    result file:filepath
    output:
    n_label:numble of label
    data_vec: the probabilitie of prediction in the 1000
    :return: probabilities, numble of label
    """
    f = ""
    data = ""
    temp = ""
    if tfile_type == 'bin':
        f = open(filepath, 'rb')
        data = np.fromfile(f,dtype=np.int64)
        temp = str(data[0]).strip().split(" ") 
    elif tfile_type == 'txt':
        f = open(filepath, 'r')
        data = f.readline()
        temp = data.strip().split(" ")
   
    n_label = len(temp)
    data_vec = np.zeros((n_label), dtype=np.float32)

    for ind, prob in enumerate(temp):
        data_vec[ind] = np.float32(prob)
    return data_vec, n_label

def create_visualization_statistical_result(prediction_file_path,
                                            dest_path, dest_name,
                                            dict_img_label):
    """
    :param prediction_file_path: 推理结果路径
    :param dest_path: 后处理结果保存的json文件路径
    :param dest_name: 结果文件的名字
    :param dict_img_label: 真实标签结果，dict形式，key为图片名称，value是标签
    :return: NA
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC  # 注意根据具体业务的需要设置文件读写方式
    modes = stat.S_IWUSR | stat.S_IRUSR  # 注意根据具体业务的需要设置文件权限
    writer = os.fdopen(os.open(os.path.join(result_json_path, json_file_name), flags, modes), 'w')
    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []

    count = 0
    n_labels = ""
    count_hit = 0
    for tfile_name in os.listdir(prediction_file_path):
        temp = tfile_name.split('.')[0]
        tfile_type = tfile_name.split('.')[1]
        index = temp.rfind('_')
        img_name = temp[:index]
        if (temp[index+1:] == '0'):
            count += 1
            filepath = os.path.join(prediction_file_path, tfile_name)
            ret = load_statistical_predict_result(filepath, tfile_type)
            prediction = np.array(ret[0]).astype(np.int16)[0] - 1
            n_labels = ret[1]
            gt = dict_img_label[img_name]
            if (n_labels == 1000):
                realLabel = int(gt)
            elif (n_labels == 1001):
                realLabel = int(gt) + 1
            else:
                realLabel = int(gt)
        else:
            continue

        if (str(realLabel) == str(prediction)):
            count_hit += 1

    if 'value' not in table_dict.keys():
        print("the item value does not exist!")
    else:
        table_dict["value"].extend(
            [{"key": "Number of images", "value": str(count)},
             {"key": "Number of classes", "value": str(n_labels)}])
        if count == 0:
            accuracy = 0
        else:
            accuracy = count_hit / count
        table_dict["value"].append({"key": " accuracy",
                                    "value": str(round(accuracy * 100, 2)) + '%'})

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

    if not (os.path.exists(annotation_file_path)):
        print("Ground truth file does not exist.")

    if not (os.path.exists(result_json_path)):
        print("Result folder doesn't exist.")

    image_label_dict = cre_groundtruth_dict_fromtxt(annotation_file_path)
    create_visualization_statistical_result(infer_result_path,
                                            result_json_path, json_file_name,
                                            image_label_dict)

    elapsed = (time.time() - start)
    print("Time used:", elapsed)