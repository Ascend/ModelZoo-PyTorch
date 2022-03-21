# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License-NC
# See LICENSE.txt for details
#
# Author: Zheng Tang (tangzhengthomas@gmail.com)
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, print_function
import sys, os

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + '/..'
)

try:
    from eval_lib.cython_eval import eval_market1501_wrap
except ImportError:
    print("Error: eval.pyx not compiled, please do 'make' before running 'python test.py'. exit")
    sys.exit()

from eval_metrics import eval_market1501
import numpy as np
import time

num_q = 100
num_g = 1000

distmat = np.random.rand(num_q, num_g) * 20
q_pids = np.random.randint(0, num_q, size=num_q)
g_pids = np.random.randint(0, num_g, size=num_g)
q_camids = np.random.randint(0, 5, size=num_q)
g_camids = np.random.randint(0, 5, size=num_g)

end = time.time()
cmc, mAP = eval_market1501_wrap(distmat,
                                q_pids,
                                g_pids,
                                q_camids,
                                g_camids, 10)
elapsed_cython = time.time() - end
print("=> Cython evaluation")
print("consume time {:.5f} \n mAP is {} \n cmc is {}".format(elapsed_cython, mAP, cmc))

end = time.time()
cmc, mAP = eval_market1501(distmat,
                           q_pids,
                           g_pids,
                           q_camids,
                           g_camids, 10)
elapsed_python = time.time() - end
print("=> Python evaluation")
print("consume time {:.5f} \n mAP is {} \n cmc is {}".format(elapsed_python, mAP, cmc))

xtimes = elapsed_python / elapsed_cython
print("=> Conclusion: cython is {:.2f}x faster than python".format(xtimes))
