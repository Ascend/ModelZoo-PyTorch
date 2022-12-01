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
sys.path.append('./reid-strong-baseline')
import os
import argparse
import glob
import re
import numpy as np
import torch
from utils.reid_metric import R1_mAP


def get_pred_label(label_dir, pre_dir):
    img_paths = glob.glob(os.path.join(label_dir, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    outputs = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        camid -= 1  # index starts from 0

        filename = img_path.split("/")[-1]
        if filename[-8:] == ".jpg.jpg":
            bin_file = filename[:-8] + "_0.bin"
        else:
            bin_file = filename[:-4] + "_0.bin"
        output = np.fromfile(os.path.join(pre_dir, bin_file), dtype="float32")
        output = torch.from_numpy(output)
        output = output.unsqueeze(0)

        pid = torch.from_numpy(np.array([pid,]))
        camid = torch.from_numpy(np.array([camid,]))
        outputs.append((output, pid, camid))

    return outputs

def eval(query_dir, gallery_dir, pred_dir):

    query = get_pred_label(query_dir, pred_dir)
    gallery = get_pred_label(gallery_dir, pred_dir)
    outputs = query + gallery

    num_query = 3368
    eval = R1_mAP(num_query, max_rank=50, feat_norm="yes")
    eval.reset()
    for output in outputs:
        eval.update(output)
    cmc, mAP = eval.compute()
    print('Validation Results')
    print("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dir", default="./data/market1501/query")
    parser.add_argument("--gallery_dir", default="./data/market1501/bounding_box_test")
    parser.add_argument("--pred_dir", default="./result/dumpOutput_device0/")
    args = parser.parse_args()
    eval(args.query_dir, args.gallery_dir, args.pred_dir)
