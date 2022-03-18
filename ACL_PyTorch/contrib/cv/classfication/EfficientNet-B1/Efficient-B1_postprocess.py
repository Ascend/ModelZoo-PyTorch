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

import sys
import os
import argparse
import re
import numpy
import json

def rename(data_dir, pre_dir):
    txtfile_2_class = dict()
    for classname in os.listdir(data_dir):
        for imgname in os.listdir(os.path.join(data_dir, classname)):
            txtfile_2_class[os.path.splitext(imgname)[0].split("_")[2]] = classname
    omoutput_txts = os.listdir(pre_dir)
    for omtxt in omoutput_txts:
        if omtxt.split("_")[0] not in txtfile_2_class.values():
        	os.rename(os.path.join(pre_dir, omtxt), os.path.join(pre_dir, txtfile_2_class.get(omtxt.split("_")[2]) + "_" + omtxt))

def classification(data_path):
    files = os.listdir(data_path)
    class_ids = sorted(f for f in files if re.match(r"^n[0-9]+$", f))
    return class_ids


def eval(data_dir, pred_dir, save_file):
    txtfiles = os.listdir(pred_dir)
    top1_acc = 0
    top5_acc = 0
    for txtfile in txtfiles:
        print("loading {}".format(txtfile))
        pre_num = numpy.loadtxt(os.path.join(pred_dir, txtfile))
        class_ids = classification(data_dir)
        class_pres = zip(class_ids, pre_num)
        class_pres_dict = dict((class_id, value) for class_id, value in class_pres)
        class_sort = max(class_pres_dict.items(), key=lambda x: x[1])
        if txtfile.split('_')[0] == class_sort[0]:
            top1_acc = top1_acc + 1
        iteams = sorted(class_pres_dict.items(), key=lambda x: x[1])
        if txtfile.split('_')[0] in [iteams[999][0], iteams[998][0], iteams[997][0], iteams[996][0], iteams[995][0]]:
            top5_acc = top5_acc + 1

    topn_acc = dict()
    topn_acc['Top1 accuracy'] = str(top1_acc / 50000 * 100) + "%"
    topn_acc['Top5 accuracy'] = str(top5_acc / 50000 * 100) + "%"
    print(topn_acc['Top1 accuracy'])
    print(topn_acc['Top5 accuracy'])
    writer = open(save_file, 'w')
    json.dump(topn_acc, writer)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./imagenet/val")
    parser.add_argument("--pre_dir", default="./result/dumpOutput_device0/")
    parser.add_argument("--save_file", default="./result.json")
    args = parser.parse_args()
    rename(args.data_dir, args.pre_dir)
    eval(args.data_dir, args.pre_dir, args.save_file)
