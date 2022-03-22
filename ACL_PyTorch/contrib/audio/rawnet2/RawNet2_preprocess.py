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


import os
import numpy as np
import argparse
import sys
sys.path.append('RawNet/python/RawNet2/')
from dataloader import TA_Dataset_VoxCeleb2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='dataset path', default="/root/datasets/VoxCeleb1/")
    parser.add_argument('--batch_size', help='batch size', default=1)
    parser.add_argument('--output', help='out bin path', default="bin_out_bs1/")
    args = parser.parse_args()
    base_dir = args.input
    out_dir = args.output
    batch_size = int(args.batch_size)

    def get_utt_list(src_dir):
        l_utt = []
        for path, dirs, files in os.walk(src_dir):
            path = path.replace('\\', '/')
            base = '/'.join(path.split('/')[-2:]) + '/'
            for f in files:
                if f[-3:] != 'wav':
                    continue
                l_utt.append(base + f)
        return l_utt

    l_val = sorted(get_utt_list(base_dir))
    TA_evalset = TA_Dataset_VoxCeleb2(list_IDs=l_val,
                                      return_label=True,
                                      window_size=11810,
                                      nb_samp=59049,
                                      base_dir=base_dir)
    if batch_size == 1:
        for item in TA_evalset:
            n = 0
            for i in item[0]:
                i.tofile(out_dir + item[1].replace('/', '$') + "$" + str(n) + ".bin")
                n += 1
    else:
        bs16_key = open('bs16_key.txt', mode='w')
        bs16 = []
        n = 0
        i = 0
        for item in TA_evalset:
            l = 0
            for t in item[0]:
                bs16_key.write(item[1].replace('/', '$') + "$" + str(n) + ".bin" + "$" + str(l) + "\n")
                l += 1
                n += 1
                bs16.append(t)
                if n == 16:
                    np.vstack(bs16).tofile(out_dir + str(i) + ".bin")
                    i += 1
                    bs16 = []
                    n = 0
        if n % 16 == 0:
            return
        for j in range(16 - (n % 16)):
            bs16_key.write("temp$" + str(j) + "\n")
            bs16.append(np.empty((59049,), dtype='float32'))
        bs16_key.close()
        np.vstack(bs16).tofile(out_dir + str(i) + ".bin")


if __name__ == '__main__':
    main()
