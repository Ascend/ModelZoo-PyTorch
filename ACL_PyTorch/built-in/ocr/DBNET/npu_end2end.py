
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8

import os
import time
import argparse
from tqdm import tqdm

import numpy as np

from ais_bench.infer.interface import InferSession

def gen_data(data_path):
    imgs = os.listdir(data_path)
    for img in imgs:
        yield np.load(os.path.join(data_path,img)), img

def build_session(device_id, om_path):
    session_npu = InferSession(device_id, om_path)
    return session_npu

def main(data_path, session, output):

    datas = gen_data(data_path)
   
    tt = 0
    num_data = 0
    for data, name in tqdm(datas):
    
        t0 = time.time()
        result = session.infer([data], "dymshape", custom_sizes=11655609)
        tt += (time.time() - t0)
        save_path = os.path.join(output, f'{name[:-4]}_0.bin')
        result[0].tofile(save_path)
        num_data += 1

        
    print('*'*50)
    print(f"NPU E2E time(s): {tt} s")
    print(f"NPU fps: {num_data / tt} fps")
    print('*'*50)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='infer E2E')
    parser.add_argument("--data_path", default="./prep_dataset", help='data path')
    parser.add_argument("--device_id", type=int, default=0, help='device id')
    parser.add_argument("--om_path", default="./db_dym_linux_x86_64.om", help='om path')
    parser.add_argument("--output", default="./npu_result", help='result save path')
    
    flags = parser.parse_args()
    if not os.path.exists(flags.output):
        os.mkdir(flags.output)

    npu_sess = build_session(flags.device_id, flags.om_path)
    main(flags.data_path, npu_sess, flags.output)
