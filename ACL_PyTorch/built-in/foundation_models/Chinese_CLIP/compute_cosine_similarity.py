# Copyright 2023 Huawei Technologies Co., Ltd
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
import argparse
import torch
import numpy as np
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_npy_path', type=str, default='./onnx_npy_path', help='path to onnx_npy_path.')
    parser.add_argument('--om_npy_path', type=str, default='./dst', help='path to om_npy_path.')
    args = parser.parse_args()
 

    cos_result = []
    for npy_path_1, npy_path_2 in zip(os.listdir(args.onnx_npy_path), os.listdir(args.om_npy_path)):
        onnx_infer = np.load(os.path.join(args.onnx_npy_path, npy_path_1))
        om_infer = np.load(os.path.join(args.om_npy_path, npy_path_2))
        cos = F.cosine_similarity(torch.from_numpy(onnx_infer), torch.from_numpy(om_infer))
        cos_result.append(cos)
    cos_average = sum(cos_result)/len(cos_result)
    print("cosine_similarity:", cos_average)
