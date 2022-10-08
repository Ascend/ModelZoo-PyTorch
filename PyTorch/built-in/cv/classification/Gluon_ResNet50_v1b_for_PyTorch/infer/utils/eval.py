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

import numpy as np
import sys
import os
import json

def cre_groundtruth_dict_fromtxt(gtfile_path):
    """
    :param filename: file contains the imagename and label number
    :return: dictionary key imagename, value is label number
    """
    img_gt_dict = {}
    with open(gtfile_path, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split()
            imgName = temp[0].split(".")[0]
            if len(temp[1]) == 0:
                print(temp[0])
                continue
            else:
                imgLab = temp[1]
            img_gt_dict[imgName] = imgLab
    return img_gt_dict


def run():
    # infer result
    infer_reault = sys.argv[1]
    real_label_path = sys.argv[2]
    json_file_name = sys.argv[3]
    topn = 5
    count_hit = np.zeros(topn)
    img_gt_dict = cre_groundtruth_dict_fromtxt(real_label_path)
    writer = open(json_file_name, 'w')
    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []
    count = 0
    with open(infer_reault, 'r')as f:
        for line in f.readlines():
            prediction = []
            temp = line.strip().split(" ")
            img_name = temp[0]
            print(img_name, "====", count)
            count += 1
            prediction = np.array([float(temp[i]) for i in range(1, len(temp))])
            n_labels = len(prediction)
            gt = img_gt_dict[img_name]
            sort_index = np.argsort(-prediction)
            if (n_labels == 1000):
                realLabel = int(gt)
            elif (n_labels == 1001):
                realLabel = int(gt) + 1
            else:
                realLabel = int(gt)
            resCnt = min(len(sort_index), topn)
            for i in range(resCnt):
                if (str(realLabel) == str(sort_index[i])):
                    count_hit[i] += 1
                    break
        print(count)
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
    run()
