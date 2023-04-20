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
import numpy as np
import argparse
import json
from tqdm import tqdm


def evaluate(args):
    input_dir = args.input_dir
    save_path = args.save_path
    label_path = args.label_path
    dtype = args.dtype

    # load predict results && gt labels
    label_result = dict()
    with open(label_path, 'r') as f:
        for label_info in f.readlines():
            image_name, label_id = label_info.split(' ')
            label_result[os.path.splitext(image_name)[0]] = np.array(int(label_id))
    predict_result = dict()
    predict_files = os.listdir(input_dir)
    predict_files = list(
        filter(lambda x:os.path.splitext(x)[1] in [".bin", ".npy"], predict_files))
    for predict_name in tqdm(predict_files):
        predict_path = os.path.join(input_dir, predict_name)
        if os.path.splitext(predict_name)[1] == ".bin":
            predict_data = np.argsort(-1 * np.fromfile(predict_path, dtype=dtype))
        else:
            predict_data = np.argsort(-1 * np.load(predict_path).reshape(-1))
        predict_result[os.path.splitext(predict_name)[0][:-2]] = {
            "top1": predict_data[0], "top5": predict_data[:5]}

    # calculate acc
    total_num = len(label_result)
    if len(predict_result) != total_num:
        raise ValueError(
            "Num of predict results not equal to num of gt results: {} != {}".format(
                len(predict_result), total_num
            ))
    num_acc1 = 0
    num_acc5 = 0
    for file_name in tqdm(predict_result):
        gt_label = label_result.get(file_name)
        predict_acc1 = predict_result.get(file_name)["top1"]
        predict_acc5 = predict_result.get(file_name)["top5"]
        num_acc1 += np.sum(predict_acc1 == gt_label)
        num_acc5 += np.sum(predict_acc5 == gt_label.repeat(5))

    # dump output data
    out_result = {
        "Top1 Acc": "{:.2f}%".format(num_acc1 * 100 / total_num),
        "Top5 Acc": "{:.2f}%".format(num_acc5 * 100 / total_num)
    }
    print(out_result)
    with open(save_path, 'w') as f:
        json.dump(out_result, f, ensure_ascii=False, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(description='SwinTransformer postprocess.')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='result dir for swintransformer model')
    parser.add_argument('-l', '--label_path', type=str, required=True,
                        help='file path for val label')
    parser.add_argument('-s', '--save_path', type=str, default='./result.json',
                        help='save path for evaluation result')
    parser.add_argument('-d', '--dtype', type=str, default='float32',
                        help='dtype for predict result')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    evaluate(args)
