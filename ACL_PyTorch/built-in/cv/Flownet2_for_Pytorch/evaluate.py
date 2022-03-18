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
import argparse
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='./data_preprocessed_bs1/gt')
parser.add_argument('--output_path', type=str, default='./output_bs1/')
parser.add_argument('-b', '--batch_size', type=int, default=1)
args = parser.parse_args()


if __name__ == '__main__':
    prediction_dir = args.output_path
    prediction = os.path.join(prediction_dir, os.listdir(prediction_dir)[0])

    gt = args.gt_path
    num_samples = len(os.listdir(gt))

    res = []
    outs = []

    pred_idx = 0
    for label_idx in tqdm(range(num_samples)):
        label_path = os.path.join(gt, '{}.bin'.format(label_idx))
        label = np.fromfile(label_path, dtype=np.float32)
        if pred_idx < label_idx // args.batch_size + 1:
            pred_idx += 1
            out_path = os.path.join(prediction, '{}_output_0.bin'.format(pred_idx - 1))
            if not os.path.exists(out_path):
                print("Error: {} not exists".format(out_path))
                continue
            out = np.fromfile(out_path, dtype=np.float32)
            # transpose for onnx
            outs = np.transpose(out.reshape(args.batch_size, 2, 448, 1024), (0, 2, 3, 1))

        out = outs[(label_idx % args.batch_size)]
        label = label.reshape(448, 1024, 2)
        label = label[:436, :, :]
        out = out[:436, :, :]
        re = np.mean(np.linalg.norm(label - out, axis=-1))
        res.append(re)

    print('{} samples, Average EPE = {}'.format(len(res), np.mean(res)))
