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

import argparse
import time
import os
from tqdm import tqdm

import numpy as np
import onnxruntime as ort

from ais_bench.infer.interface import InferSession



def com_cos(p1, p2):
    vec1 = p1.reshape(1,-1)
    vec2 = p2.reshape(1,-1)
    return vec1.dot(vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main(data_path, onnx_path, npu_session, onnx_session):

    # onnxruntime
    input_name0 = onnx_session.get_inputs()[0].name

    imgs = os.listdir(data_path)
    datas = [[np.load(os.path.join(data_path,img))] for img in imgs]

    t = 0
    cos_list = []
    for data in tqdm(datas):
        onnx_result = ort_session.run(None, {input_name0:data[0]})
        t0 = time.time()
        npu_result = npu_session.infer(data, "dymshape", custom_sizes=1655609)
        t += (time.time() - t0)
        cos_list.append(com_cos(onnx_result[0], npu_result[0]))        

    print("*"*50)
    print(f'E2E NPU time(s): {t}s')
    print(f'the result cos is {np.mean(cos_list)}')
    print("*"*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='infer E2E')
    parser.add_argument("--data_path", default="./pre_npy", help='data path')
    parser.add_argument("--onnx_path", default="./dbnet_fix.onnx", help='onnx path')
    parser.add_argument("--device", default=0, type=int, help='npu device')
    parser.add_argument("--om_path", default="./om/dbnet_fix_linux_x86_64.om", help='om path')
    flags = parser.parse_args()
    
    db_session = InferSession(flags.device, flags.om_path)
    ort_session = ort.InferenceSession(flags.onnx_path)

    main(flags.data_path, flags.onnx_path, db_session, ort_session)
