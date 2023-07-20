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

def postprocess(output_dir, label_path):

    correct = 0
    total = 0
    with open(label_path, 'r') as label_f:
        for line in tqdm(label_f.readlines()):
            img_name, label = line.strip().split(' ')
            img_name = img_name.split('.')[0]
            labels = [int(label)]
            f = f'{img_name}_0.bin'
            f_path = os.path.join(output_dir, f)
            out = torch.tensor(np.fromfile(f_path, dtype='float32').reshape(-1, 1000))
            probabilities = torch.nn.functional.softmax(out, dim=1)
            pred = torch.argmax(probabilities, 1)
            correct += torch.sum(pred[:len(labels)] == torch.tensor(labels))
            total += len(labels)

    print('Top 1 acc: ', float(correct / total))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EfficientNetV2 postprocess')
    parser.add_argument('--output_dir', type=str, help='om inference output directory', required=True)
    parser.add_argument('--label_path', type=str, help='path of the label file', required=True)
    args = parser.parse_args()

    postprocess(args.output_dir, args.label_path)