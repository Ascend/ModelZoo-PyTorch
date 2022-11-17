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
import sys
import argparse
from pathlib import Path
import numpy as np
sys.path.append(r"./3DMPPE_ROOTNET_RELEASE")
from data.MuPoTS.MuPoTS import MuPoTS

def evaluate(result_path, result_file, img_path, ann_path):
    print('postprocessing')
    bin_path = os.listdir(result_path)[0]
    result_path = os.path.join(result_path, bin_path)
    bin_list = os.listdir(result_path)
    bin_list.sort(key=lambda x: int(x[:-6]))
    preds = []
    for i, f in enumerate(bin_list):
        bin_path = os.path.join(result_path, f)
        coord_out = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)
        preds.append(coord_out)
    # evaluate
    preds = np.concatenate(preds, axis=0)
    testset = MuPoTS('test', img_path, ann_path)
    if not os.path.exists(result_file):
        os.makedirs(result_file)
    testset.evaluate(preds, result_file)
    print('postprocess finised')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of 3D-ResNets')
    parser.add_argument('--img_path', default='MuPoTS/MultiPersonTestSet', type=Path, help='Directory path of videos')
    parser.add_argument('--ann_path', default='MuPoTS/MuPoTS-3D.json', type=Path, help='Annotation file path')
    parser.add_argument('--input_path', default='out_bs1', type=Path, help='Directory path of videos')
    parser.add_argument('--result_file', default='result_bs1', type=Path, help='Directory path of binary output data')
    opt = parser.parse_args()
    evaluate(opt.input_path, opt.result_file, opt.img_path, opt.ann_path)