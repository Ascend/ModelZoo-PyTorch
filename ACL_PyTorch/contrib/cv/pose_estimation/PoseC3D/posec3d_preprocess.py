# Copyright (c) OpenMMLab. All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from mmcv import Config
import torch.nn.functional as F
from mmaction.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset HMDB51 Preprocessing')
    parser.add_argument('--config',
                        default='./mmaction2/configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.py',
                        help='config file path')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
    parser.add_argument('--num_worker', default=8, type=int, help='Number of workers for inference')
    parser.add_argument('--data_root', default='/opt/npu/hmdb51/rawframes/', type=str)
    parser.add_argument('--ann_file', default='/opt/npu/hmdb51/hmdb51.pkl', type=str)
    parser.add_argument('--name', default='./prep_hmdb51_bs1', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.data_prefix = args.data_root

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=args.batch_size,
        workers_per_gpu=args.num_worker,
        dist=False,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    root_path = os.path.dirname(args.ann_file)
    out_path = args.name
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    info_file = open(os.path.join(root_path, 'hmdb51.info'), 'w')

    pbar = tqdm(data_loader)
    i = 0
    for data in pbar:
        pbar.set_description('Preprocessing ')
        imgs = data['imgs']
        label = data['label']

        for batch in range(imgs.shape[0]):
            l = label.cpu().numpy()[batch]
            info_file.write(str(args.batch_size*i+batch) + ' ' + str(l))
            info_file.write('\n')

        if imgs.shape[0] != args.batch_size:
            imgs = F.pad(imgs, (0,0,0,0,0,0,0,0,0,args.batch_size-imgs.shape[0]))

        bin_info = imgs.cpu().numpy()
        bin_info.tofile(out_path + '/' + str(i) + '.bin')
        i += 1

if __name__ == '__main__':
    main()