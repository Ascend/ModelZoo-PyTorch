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
This script aims to automate the process of calculating average results
stored in the test.log files over multiple splits.

How to use:
For example, you have done evaluation over 20 splits on VIPeR, leading to
the following file structure

log/
    eval_viper/
        split_0/
            test.log-xxxx
        split_1/
            test.log-xxxx
        split_2/
            test.log-xxxx
    ...

You can run the following command in your terminal to get the average performance:
$ python tools/parse_test_res.py log/eval_viper
"""
import os
import re
import glob
import numpy as np
import argparse
from collections import defaultdict

from torchreid.utils import check_isfile, listdir_nohidden


def parse_file(filepath, regex_mAP, regex_r1, regex_r5, regex_r10, regex_r20):
    results = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()

            match_mAP = regex_mAP.search(line)
            if match_mAP:
                mAP = float(match_mAP.group(1))
                results['mAP'] = mAP

            match_r1 = regex_r1.search(line)
            if match_r1:
                r1 = float(match_r1.group(1))
                results['r1'] = r1

            match_r5 = regex_r5.search(line)
            if match_r5:
                r5 = float(match_r5.group(1))
                results['r5'] = r5

            match_r10 = regex_r10.search(line)
            if match_r10:
                r10 = float(match_r10.group(1))
                results['r10'] = r10

            match_r20 = regex_r20.search(line)
            if match_r20:
                r20 = float(match_r20.group(1))
                results['r20'] = r20

    return results


def main(args):
    regex_mAP = re.compile(r'mAP: ([\.\deE+-]+)%')
    regex_r1 = re.compile(r'Rank-1  : ([\.\deE+-]+)%')
    regex_r5 = re.compile(r'Rank-5  : ([\.\deE+-]+)%')
    regex_r10 = re.compile(r'Rank-10 : ([\.\deE+-]+)%')
    regex_r20 = re.compile(r'Rank-20 : ([\.\deE+-]+)%')

    final_res = defaultdict(list)

    directories = listdir_nohidden(args.directory, sort=True)
    num_dirs = len(directories)
    for directory in directories:
        fullpath = os.path.join(args.directory, directory)
        filepath = glob.glob(os.path.join(fullpath, 'test.log*'))[0]
        check_isfile(filepath)
        print(f'Parsing {filepath}')
        res = parse_file(
            filepath, regex_mAP, regex_r1, regex_r5, regex_r10, regex_r20
        )
        for key, value in res.items():
            final_res[key].append(value)

    print('Finished parsing')
    print(f'The average results over {num_dirs} splits are shown below')

    for key, values in final_res.items():
        mean_val = np.mean(values)
        print(f'{key}: {mean_val:.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='Path to directory')
    args = parser.parse_args()
    main(args)
