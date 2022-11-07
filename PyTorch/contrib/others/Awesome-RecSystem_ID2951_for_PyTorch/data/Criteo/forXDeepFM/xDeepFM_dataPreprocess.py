# BSD 3-Clause License
#
# Copyright (c) 2022 Huawei Technologies Co., Ltd
# All rights reserved.
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

import os
import numpy
import shutil
import pickle
import pandas as pd
import csv
import math
from collections import defaultdict
from data.Criteo.util import get_raw_data, split_data

EACH_FILE_DATA_NUM = 204800

"""
[1] PaddlePaddle implemantation of DeepFM for CTR prediction
    https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/ctr/deepfm/data/preprocess.py
"""

def scan_train_txt(filename):
    feat_cnt = defaultdict(lambda: 0)
    count = 0
    for row in csv.DictReader(open(filename)):
        count += 1
        # if count >= EACH_FILE_DATA_NUM * 20:
        #     break
        for key, val in row.items():
            if 'C' in key:
                if val == '':
                    feat_cnt[str(key) + '#' + 'absence'] += 1
                else:
                    feat_cnt[str(key) + '#' + str(val)] += 1
    return feat_cnt


def get_feat(filename, feat_cnt):
    T = 4
    featSet = set()
    count = 0
    for row in csv.DictReader(open(filename)):
        count += 1
        if count % EACH_FILE_DATA_NUM == 0:
            print('generating feature dict', count / 45000000)

        for key, val in row.items():
            if 'I' in key and key != "Id":
                if val == '':
                    featSet.add(str(key) + '#' + 'absence')
                else:
                    val = int(float(val))
                    if val > 2:
                        val = int(math.log(float(val)) ** 2)
                    else:
                        val = 'SP' + str(val)
                    featSet.add(str(key) + '#' + str(val))
                continue
            if 'C' in key:
                if val == '':
                    feat = str(key) + '#' + 'absence'
                else:
                    feat = str(key) + '#' + str(val)
                if feat_cnt[feat] > T:
                    featSet.add(feat)
                else:
                    featSet.add(str(key) + '#' + str(feat_cnt[feat]))
                continue

    featIndex = dict()
    for index, feat in enumerate(featSet, start=1):
        featIndex[feat] = index
    print('feat dict num:', len(featIndex))

    return featIndex


if __name__ == '__main__':
    train_csv_file = '../input.csv'
    if not os.path.isdir('train_data'):
        os.mkdir('train_data')
    if not os.path.isdir('test_data'):
        os.mkdir('test_data')
    if not os.path.isdir('aid_data'):
        os.mkdir('aid_data')

    get_raw_data()
    split_data()

    reader = pd.read_csv('../train.txt', delimiter='\t', header=None, chunksize=1024 * 1024)
    for idx, r in enumerate(reader):
        if idx == 0:
            r.columns = ['Label'] + ['I' + str(i) for i in range(1, 14)] + ['C' + str(i) for i in range(1, 27)]
            r.to_csv(train_csv_file, mode='a', header=True, index=False)
        else:
            r.to_csv(train_csv_file, mode='a', header=False, index=False)

    feat_cnt = scan_train_txt(train_csv_file)
    featIndex = get_feat(train_csv_file, feat_cnt)

    # Save dictionary
    freq_ = 4
    dir_feat_dict_ = 'aid_data/feat_dict_' + str(freq_) + '.pkl2'
    with open(dir_feat_dict_, 'wb') as fout:
        pickle.dump(featIndex, fout)

    feat_cnt_dict_ = 'aid_data/feat_cnt_' + str(freq_) + '.pkl2'
    feat_cnt_dict = {}
    for k, v in feat_cnt.items():
        feat_cnt_dict[k] = v
    with open(feat_cnt_dict_, 'wb') as fout:
        pickle.dump(feat_cnt_dict, fout)
    print('args.num_feat ', len(featIndex) + 1)
    print('Done!')

