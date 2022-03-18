#
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
#
import sys
import os.path
import glob
import pickle
import lmdb
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.progress_bar import ProgressBar
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))

# configurations
img_folder = '../../datasets/DIV2K800/DIV2K800/*'  # glob matching pattern
lmdb_save_path = '../../datasets/DIV2K800/DIV2K800.lmdb'  # must end with .lmdb

img_list = sorted(glob.glob(img_folder))
dataset = []
data_size = 0

print('Read images...')
pbar = ProgressBar(len(img_list))
for i, v in enumerate(img_list):
    pbar.update('Read {}'.format(v))
    img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
    dataset.append(img)
    data_size += img.nbytes
env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

pbar = ProgressBar(len(img_list))
with env.begin(write=True) as txn:  # txn is a Transaction object
    for i, v in enumerate(img_list):
        pbar.update('Write {}'.format(v))
        base_name = os.path.splitext(os.path.basename(v))[0]
        key = base_name.encode('ascii')
        data = dataset[i]
        if dataset[i].ndim == 2:
            H, W = dataset[i].shape
            C = 1
        else:
            H, W, C = dataset[i].shape
        meta_key = (base_name + '.meta').encode('ascii')
        meta = '{:d}, {:d}, {:d}'.format(H, W, C)
        # The encode is only essential in Python 3
        txn.put(key, data)
        txn.put(meta_key, meta.encode('ascii'))
print('Finish writing lmdb.')

# create keys cache
keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    print('Create lmdb keys cache: {}'.format(keys_cache_file))
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    pickle.dump(keys, open(keys_cache_file, "wb"))
print('Finish creating lmdb keys cache.')
