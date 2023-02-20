# Copyright 2023 Huawei Technologies Co., Ltd
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
import sys
import json
import time
import argparse
from tqdm import tqdm
import numpy as np


#============================================================================
# Variables
#============================================================================
np.set_printoptions(threshold=sys.maxsize)

LABEL_FILE = "HiAI_label.json"


#============================================================================
# Functions
#============================================================================
def gen_file_name(img_name):
    full_name = img_name.split('/')[-1]
    index = full_name.rfind('.')
    return full_name[:index]


def cre_groundtruth_dict(gtfile_path):
    img_gt_dict = {}
    for gtfile in os.listdir(gtfile_path):
        if (gtfile != LABEL_FILE):
            with open(os.path.join(gtfile_path, gtfile), 'r') as f:
                gt = json.load(f)
                ret = gt["image"]["annotations"][0]["category_id"]
                img_gt_dict[gen_file_name(gtfile)] = ret
    return img_gt_dict


def cre_groundtruth_dict_fromtxt(gtfile_path):
    img_count = 0
    img_gt_dict = {}
    with open(gtfile_path, 'r')as f:
        for line in f.readlines():
            img_count += 1
            temp = line.strip().split(" ")
            imgName = temp[0].split(".")[0]
            imgLab = temp[1]
            img_gt_dict[imgName] = imgLab
    return img_gt_dict, img_count


def load_statistical_predict_result(filepath, index):
    """
    the prediction esult file data extraction
    """

    with open(filepath, 'r') as f:
        for i, index_data in enumerate(f):
            if i == index:
                data = index_data
        temp = data.strip().split(" ")
        n_label = len(temp)
        if data == '':
            n_label = 0
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
                                            result_store_path, json_file_name,
                                            batch_size, img_gt_dict,
                                            img_num, topn=5):
    writer = open(os.path.join(result_store_path, json_file_name), 'w')
    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []

    count = 0
    resCnt = 0
    n_labels = 0
    count_hit = np.zeros(topn)
    print(prediction_file_path)
    for tfile_name in tqdm(os.listdir(prediction_file_path)):
        for i in range(batch_size):
            count += 1
            temp = tfile_name.split('.')[0]
            index = temp.find('_') + 1
            img_index = temp[index: index+5]
            convert_index = int(img_index) * batch_size + i + 1
            if convert_index > img_num:
                break
            img_name = "ILSVRC2012_val_{:08d}".format(convert_index)
            filepath = os.path.join(prediction_file_path, tfile_name)
            ret = load_statistical_predict_result(filepath, i)
            prediction = ret[0]
            n_labels = ret[1]
            sort_index = np.argsort(-prediction)
            gt = img_gt_dict[img_name]
            if (n_labels == 1000):
                realLabel = int(gt)
            elif (n_labels == 1001):
                realLabel = int(gt) + 1
            else:
                realLabel = int(gt)

            resCnt = min(len(sort_index), topn)
            for j in range(resCnt):
                if (str(realLabel) == str(sort_index[j])):
                    count_hit[j] += 1
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
                                        "value": str(round(accuracy[i] * 100, 2)) + '%'})
        json.dump(table_dict, writer)
    writer.close()


#============================================================================
# Main
#============================================================================
if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder-davinci-target', type=str, default="./result/outputs_bs1_om/")
    parser.add_argument('--annotation-file-path', type=str, default="/opt/npu/imageNet/val_label.txt")
    parser.add_argument('--result-json-path', type=str, default="./result")
    parser.add_argument('--json-file-name', type=str, default="result_bs1.json")
    parser.add_argument('--batch-size', type=int, default=1)

    opt = parser.parse_args()

    sub_folder = os.listdir(opt.folder_davinci_target)
    opt.folder_davinci_target = os.path.join(opt.folder_davinci_target, sub_folder[0])

    img_label_dict, img_num = cre_groundtruth_dict_fromtxt(opt.annotation_file_path)
    create_visualization_statistical_result(opt.folder_davinci_target,
                                            opt.result_json_path,
                                            opt.json_file_name,
                                            opt.batch_size,
                                            img_label_dict,
                                            img_num, topn=1)

    elapsed = (time.time() - start)
    print("Time used:", elapsed)
