# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import sys
sys.path.append(os.getcwd())
import argparse
from siamfc import train, train_dist

npu_id = 0

data_dir = './data/ILSVRC_VID_CURATION'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=" SiamFC Train")
    parser.add_argument('--data', default=data_dir, type=str, help=" the path of data")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--workers', default=24, type=int)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    if args.world_size == 1:
        # 1P
        train(args.data, args.workers, args.epoch)
    else:
        # 8P
        train_dist(args)
