# Copyright 2022 Huawei Technologies Co., Ltd
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
import sys
import os

import torch
import numpy as np
from tqdm import tqdm

sys.path.append(r"./FLAVR")
import myutils
from FLAVR.dataset.ucf101_test import get_loader

def parse_args():
    parser = argparse.ArgumentParser(description='compute metircs')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='the data root of test dataset')
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--n_outputs', type=int, default=3,
                        help='For Kx FLAVR, use n_outputs k-1')
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()
    return args

def postprocess(data_dir, result_dir, n_outputs):
    _, psnrs, ssims = myutils.init_meters('1*L1')
    test_loader = get_loader(data_dir, batch_size=1, shuffle=False, num_workers=args.num_workers)
    for i, (_, gt) in enumerate(tqdm(test_loader)):
        out = []
        for idx in range(n_outputs):
            ret_path = os.path.join(result_dir, '{}_{}.bin'.format(i, idx))
            ret = np.fromfile(ret_path, dtype="float32")
            ret = torch.from_numpy(ret).reshape(1,3,224,224)
            out.append(ret)
        
        out = torch.cat(out)
        gt = torch.cat(gt)
        myutils.eval_metrics(out, gt, psnrs, ssims)
    
    print('PSNR: %f, SSIM: %fn' %(psnrs.avg, ssims.avg))

if __name__ == "__main__":
    args = parse_args()
    postprocess(args.data_dir, args.result_dir, args.n_outputs)