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
import sys
import argparse
import torch
sys.path.append('./GMA/core')
import numpy as np
from tqdm import tqdm
from utils.utils import InputPadder, forward_interpolate


parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='./data_preprocessed_bs1/gt')
parser.add_argument('--output_path', type=str, default='./output_bs1/')
parser.add_argument('-s', '--status', type=str, default='clean')
args = parser.parse_args()


if __name__ == '__main__':
    prediction = args.output_path

    gt = args.gt_path
    num_samples = len(os.listdir(gt))

    res = []
    outs = []
    epe_list = []
    pred_idx = 0
    padder = InputPadder([1, 3, 436, 1024])
    for label_idx in tqdm(range(num_samples)):
        label_path = os.path.join(gt, '{}.bin'.format(label_idx))
        label = np.fromfile(label_path, dtype=np.float32).reshape(436, 1024, 2)
        label = torch.from_numpy(label).permute(2, 0, 1).float()
        pred_idx += 1
        out_path = os.path.join(prediction, '{}_0.bin'.format(pred_idx - 1))
        if not os.path.exists(out_path):
            print("Error: {} not exists".format(out_path))
            continue
        out = np.fromfile(out_path, dtype=np.float32).reshape(1, 2, 440, 1024)
        out=torch.tensor(out)
        flow = padder.unpad(out[0]).cpu()
        epe = torch.sum((flow - label)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all<1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)
    print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (args.status, epe, px1, px3, px5))

        
