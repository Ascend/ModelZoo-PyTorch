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
import sys
import time
import numpy as np


np.set_printoptions(threshold=sys.maxsize)


def cre_groundtruth_dict_fromtxt(val_label_path):
    img_label_dict = {}
    with open(val_label_path, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            img_name = temp[0].split(".")[0]
            img_lab = temp[1]
            img_label_dict[img_name] = img_lab
    return img_label_dict


def create_visualization_statistical_result(prediction_file_path,
                                            img_label_dict, topn=5):
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
        prediction = np.fromfile(filepath, dtype=np.float32)
        n_labels = len(prediction)
        sort_index = np.argsort(-prediction)
        gt = img_label_dict[img_name]
        if n_labels == 1000:
            real_label = int(gt)
        elif n_labels == 1001:
            real_label = int(gt) + 1
        else:
            real_label = int(gt)

        res_count = min(len(sort_index), topn)
        for i in range(res_count):
            if str(real_label) == str(sort_index[i]):
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
    print(table_dict)


if __name__ == '__main__':
    start = time.time()
    try:
        # infer result file path
        infer_result_path = sys.argv[1] 

        # annotation files path, "val_label.txt"
        annotation_file_path = sys.argv[2]

    except IndexError:
        print("Stopped!")
        exit(1)

    if not os.path.exists(infer_result_path):
        print("infer result path does not exist.")

    if not os.path.exists(annotation_file_path):
        print("Ground truth file does not exist.")

    img_label = cre_groundtruth_dict_fromtxt(annotation_file_path)
    create_visualization_statistical_result(infer_result_path,
                                            img_label, 5)

    elapsed = (time.time() - start)
