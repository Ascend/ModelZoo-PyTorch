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

import sys
import os
import multiprocessing
import torch
import numpy as np
import time
from tqdm import tqdm

sys.path.append('./AdvancedEAST-PyTorch')

from utils import eval_pre_rec_f1

def eval_func(i, out, gt_xy_list):
    eval_p_r_f = eval_pre_rec_f1()
    eval_p_r_f.add(out, gt_xy_list)
    mPre, mRec, mF1_score = eval_p_r_f.val()
    np.save('eval_temp/{}.npy'.format(str(i)), [mPre, mRec, mF1_score])
    eval_p_r_f.reset()


def eval(data_dir, result_dir):
    train_label_dir = os.path.join(data_dir, 'labels_3T736')
    img_list = os.listdir(result_dir)
    thread_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    i = 0
    scale = len(img_list)
    start = time.perf_counter()
    pbar = tqdm(total=len(img_list))
    update = lambda *args: pbar.update()
    for img_file in img_list:
        gt_xy_list = [np.load(os.path.join(train_label_dir, img_file[:-6] + '.npy'))]
        out = np.fromfile(os.path.join(result_dir, img_file), dtype=np.float32)
        size = 736 // 4
        out.shape = (1, 7, size, size)
        out = torch.from_numpy(out)
        thread_pool.apply_async(eval_func, args=(i, out, gt_xy_list), callback=update)
        i += 1
        
    thread_pool.close()
    thread_pool.join()

    mPre = mRec = mF1_score = 0
    size = len(img_list)
    for i in range(size):
        arr = np.load('eval_temp/{}.npy'.format(str(i)))
        mPre += arr[0]
        mRec += arr[1]
        mF1_score += arr[2]
    mPre /= size
    mRec /= size
    mF1_score /= size
    print('precision:{:.2f}% recall:{:.2f}% f1-score:{:.2f}%'.format(mPre, mRec, mF1_score))


if __name__ == '__main__':
    data_dir = sys.argv[1]
    result_dir = sys.argv[2]
    if not os.path.exists('eval_temp'):
        os.mkdir('eval_temp')
    eval(data_dir, result_dir)
