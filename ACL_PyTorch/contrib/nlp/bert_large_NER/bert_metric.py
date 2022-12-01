# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def get_single_result(result_bin_file: Path, num_classes=9, seq_len=512):
    result1 = np.load(result_bin_file)
    result = result1[0]
    pred = result.argmax(axis=1)
    return pred


def metric(result_dir='./bert-large-OUT/bs16/2022_09_28-06_23_32', anno_file='./bert_bin/bert_bin_20220928-061343.anno'):
    result_dir = Path(result_dir)
    anno_file = Path(anno_file)
    item_num, correct = 0, 0
    with anno_file.open() as annofile:
        for line in tqdm(annofile):
            line = line.strip('\n')
            idx, *tags = map(int, line.split(' '))
            tags = np.array(tags)
            pred = get_single_result(result_dir / f'input_ids_{idx}_{idx}_0.npy')[:len(tags)]  # throw prediction of padding
            
            item_num += len(tags)
            correct += (tags == pred).sum()
    acc = correct / item_num
    print(f'Accuracy: {acc * 100: .2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='./bert-large-OUT/bs16/2022_09_28-06_23_32')
    parser.add_argument('--anno_file', type=str, default='./bert_bin/bert_bin_20220928-061343.anno')
    args = parser.parse_args()

    metric(args.result_dir, args.anno_file)
