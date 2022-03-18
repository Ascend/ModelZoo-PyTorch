# encoding=utf-8
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

from __future__ import absolute_import, division, print_function

import argparse
import numpy as np


def om_post(ar):
    res_dir=ar.dump_output    
    label = np.loadtxt('albert.label')
    yes = 0
    for i in range(len(label)):
        res_path = res_dir + '/Bert_%d_1.txt' % i
        res = np.loadtxt(res_path)
        if (res[1] > res[0] and label[i] > 0.5) or (res[1] < res[0] and label[i] < 0.5):
            yes += 1
    print("acc = {:.3f}".format(yes / len(label)))

    prof_path=ar.dump_perf
    if prof_path=="":return
    with open(prof_path, 'r') as f:
        txt = f.readlines()[3]
    fps = txt.split(',')[1].split(':')[1]
    fps = float(fps) * 4
    print("fps = {:.2f} ".format(fps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_output", default="", type=str,required=True,
                        help="./result/dumpOutput_xxx, contains acc")
    parser.add_argument("--dump_perf", default="", type=str,
                        help="./result/perf_bert_xxx, contains fps.")
    ar = parser.parse_args()

    om_post(ar)


if __name__ == "__main__":
    main()
