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
Compute channel-wise mean and standard deviation of a dataset.

Usage:
$ python compute_mean_std.py DATASET_ROOT DATASET_KEY

- The first argument points to the root path where you put the datasets.
- The second argument means the specific dataset key.

For instance, your datasets are put under $DATA and you wanna
compute the statistics of Market1501, do
$ python compute_mean_std.py $DATA market1501
"""
import argparse

import torchreid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    parser.add_argument('sources', type=str)
    args = parser.parse_args()

    datamanager = torchreid.data.ImageDataManager(
        root=args.root,
        sources=args.sources,
        targets=None,
        height=256,
        width=128,
        batch_size_train=100,
        batch_size_test=100,
        transforms=None,
        norm_mean=[0., 0., 0.],
        norm_std=[1., 1., 1.],
        train_sampler='SequentialSampler'
    )
    train_loader = datamanager.train_loader

    print('Computing mean and std ...')
    mean = 0.
    std = 0.
    n_samples = 0.
    for data in train_loader:
        data = data['img']
        batch_size = data.size(0)
        data = data.view(batch_size, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        n_samples += batch_size

    mean /= n_samples
    std /= n_samples
    print('Mean: {}'.format(mean))
    print('Std: {}'.format(std))


if __name__ == '__main__':
    main()
