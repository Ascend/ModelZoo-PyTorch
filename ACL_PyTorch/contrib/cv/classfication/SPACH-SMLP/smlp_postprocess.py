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

import argparse
import numpy as np
import sys
import os
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        '验证SMLP模型推理Top1 Top5准确度', add_help=True)
    parser.add_argument('--infer_result_dir', default='~/spach-smlp/ais_infer/2022_07_09-17_36_18', type=str, metavar='MODEL',
                        help='ais_infer工具推理输出文件路径，会根据日期变动，请按照实际情况修改')
    parser.add_argument('--n', default="50000", type=int,
                        help='验证数据集大小，默认验证imagenet-val全部50000张图片')
    args = parser.parse_args()
    return args


def postprocess(args):
    infer_result_dir = args.infer_result_dir
    n = args.n
    top_k = 5
    acc_cnt = 0
    acc_cnt_top5 = 0

    for i in tqdm(range(n)):
        infer_result_path = os.path.join(
            infer_result_dir, f"sample_id_{i}_output_0.npy")
        arr = np.load(infer_result_path)[0]

        infer_label = np.argmax(arr)
        arr_topk = np.argsort(arr)

        true_label = i // 50
        if infer_label == true_label:
            acc_cnt += 1
        if true_label in arr_topk[-top_k:]:
            acc_cnt_top5 += 1
    print(f"acc1:{acc_cnt / n:.4f}, acc5:{acc_cnt_top5 / n :.4f}")

if __name__ == '__main__':
    args = get_args()
    postprocess(args)
