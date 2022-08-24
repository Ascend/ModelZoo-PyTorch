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

import torch
import numpy as np
import os
import os.path as osp
import sys
import argparse

sys.path.insert(0, "./FairMOT/src")
import lib.datasets.dataset.jde as datasets

def process(data_root, seqs, output_dir):
    for seq in seqs:
        print("process the" + osp.join(data_root, seq, 'img1') + " files")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        img_size = (1088, 608)
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), img_size)
        for i, (path, img, img0) in enumerate(dataloader):
            blob = torch.from_numpy(img).unsqueeze(0)
            blob = np.array(blob).astype(np.float32)
            blob.tofile(osp.join(output_dir, seq + "_"+"{:0>6d}".format(i)+".bin")) 

if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument("--data_root", type=str, default="./dataset")
    parse.add_argument("--output_dir", type=str, default="./pre_dataset")
    args = parse.parse_args()
    

    seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
  
    data_dir = args.data_root
    output_dir = args.output_dir
    data_root = os.path.join(data_dir, 'MOT17/images/train')
    seqs = [seq.strip() for seq in seqs_str.split()]
    process(data_root, seqs, output_dir)