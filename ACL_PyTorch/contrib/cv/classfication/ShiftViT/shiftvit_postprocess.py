# Copyright 2022 Huawei Technologies Co., Ltd
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


import argparse
import tqdm
from pathlib import Path
import numpy as np


def compute_accuracy(result_dir, gt_path):

    all_labels = np.load(gt_path)
    cnt_total = 0
    cnt_top1 = 0
    cnt_top5 = 0
    for res_path in tqdm.tqdm(Path(result_dir).iterdir()):
        if res_path.suffix != '.bin' or res_path.name.startswith('padding_'):
            continue
        batch_idx = res_path.stem.replace('_0', '').replace('batch-', '') 
        batch_idx = int(batch_idx)
        labels = all_labels[batch_idx]
        results = np.fromfile(res_path, np.float32)
        results = results.reshape(labels.size, -1)

        for pred, label in zip(results, labels):
            cnt_total += 1
            if np.argmax(results) == label:
                cnt_top1 += 1
            if label in np.argsort(pred)[-5:]:
                cnt_top5 += 1

    acc_top1 = cnt_top1 / cnt_total
    acc_top5 = cnt_top5 / cnt_total
    print(f"Acc@Top1:{acc_top1:.4f}, Acc@Top5:{acc_top5:.4f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        'Calculate accuracy based on infer results.')
    parser.add_argument('--result-dir', required=True, type=str, 
                        help='path to infer result directory.')
    parser.add_argument('--gt-path', required=True, type=str,
                        help='path to groundtruth.')
    args = parser.parse_args()

    compute_accuracy(args.result_dir, args.gt_path)

