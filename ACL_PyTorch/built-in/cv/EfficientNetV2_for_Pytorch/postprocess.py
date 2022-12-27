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

import os
import argparse

from tqdm import tqdm
import numpy as np
import torch

def postprocess(output_dir):

    correct = 0
    total = 0
    with open('label.txt', 'r') as label_f:
        for i, line in enumerate(tqdm(label_f.readlines())):
            labels = [int(i) for i in line.strip().split(' ')]
            f = f'{i}_0.bin'
            f_path = os.path.join(output_dir, f)
            out = torch.tensor(np.fromfile(f_path, dtype='float32').reshape(-1, 1000))
            probabilities = torch.nn.functional.softmax(out, dim=1)
            pred = torch.argmax(probabilities, 1)
            matches = torch.eq(pred[:len(labels)], torch.tensor(labels))
            correct += torch.sum(pred[:len(labels)] == torch.tensor(labels))
            total += len(labels)

    print('Top 1 acc: ', float(correct / total))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EfficientNetV2 postprocess')
    parser.add_argument('--output_dir', type=str, help='om inference output directory', required=True)
    args = parser.parse_args()

    postprocess(args.output_dir)