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
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Segmentron')
    parser.add_argument('--config-file', metavar="FILE",
                        help='config file path')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # for evaluation
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    # for visual
    parser.add_argument('--input-img', type=str, default='tools/demo_vis.jpg',
                        help='path to the input image or a directory of images')
    # config options
    parser.add_argument('opts', help='See config for all options',
                        default=None, nargs=argparse.REMAINDER)

    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    args = parser.parse_args()

    return args