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

import numpy as np
import argparse

parser = argparse.ArgumentParser('Calculation FPS', add_help=False)
parser.add_argument('--log_path', default='bs1_time.log')
parser.add_argument('--batch_size', default=1,type=int)
args = parser.parse_args()

weight = [0.17, 0.06, 0.53, 0.18, 0.05, 0.009, 0.0014, 0.0006, 0.005]
weight = np.array(weight)
val_times = []
with open(args.log_path, 'r') as l:
    for line in l.readlines():
        if line.startswith('Inference average time without first time: '):
            val_time = float(line.split(':')[1].replace('ms', '')) / 1000
            val_times.append(val_time)
val_times = np.array(val_times)
fps = 1 / sum(val_times * weight) * args.batch_size * 4
print(fps)