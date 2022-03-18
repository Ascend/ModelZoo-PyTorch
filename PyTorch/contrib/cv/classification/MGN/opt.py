"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="../data/Market-1501",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis', 'prof'],
                    help='train or evaluate ')

parser.add_argument('--query_image',
                    default='0001_c1s1_001051_00.jpg',
                    help='path to the image you want to query')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--weight',
                    default='weights/model_500.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    default=500,
                    type=int,
                    help='number of epoch to train')

parser.add_argument('--lr',
                    default=2e-4,
                    type=float,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[320, 380],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument("--batchid",
                    default=4,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=4,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=8,
                    help='the batch size for test')

parser.add_argument('--seed', type=int, default=1)

parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--device_num', default=-1, type=int,
                    help='device_num')
parser.add_argument('--npu', action='store_true',
                    help="npu")
parser.add_argument('--addr', default='127.0.0.1',
                type=str, help='master addr')
parser.add_argument('--finetune', action='store_true',
                    help="finetune")

opt = parser.parse_args()
