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
from __future__ import print_function
import os
import sys
import numpy as np
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
sys.path.append('./SegmenTron')
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.utils.distributed import make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from tqdm import tqdm


class Pretreatment(object):
    def __init__(self, args):
        self.args = args
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode='testval', transform=input_transform, root=self.args.datasets_input_path)
        val_sampler = make_data_sampler(val_dataset, False, False)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=12,
                                          pin_memory=True)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def process(self):
        save_path = os.path.join(self.args.datasets_output_path, 'leftImg8bit')
        save_mask_path = os.path.join(self.args.datasets_output_path, 'gtFine')
        if not os.path.exists(self.args.datasets_output_path):
            os.mkdir(self.args.datasets_output_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(save_mask_path):
            os.mkdir(save_mask_path)

        with tqdm(total=len(self.val_loader)) as pbar:
            for i, (image, target, filename) in enumerate(self.val_loader):
                pbar.set_description('Processing:' + str((i+1)))
                imgs = np.array(image).astype(np.float32)
                imgs.tofile(os.path.join(save_path, filename[0].split('.')[0] + ".bin"))
                mask = np.array(target).astype(np.float32)
                temp_path = filename[0].replace('leftImg8bit', 'gtFine_labelIds')
                mask.tofile(os.path.join(save_mask_path, temp_path.split('.')[0] + ".bin"))
                pbar.update(1)


if __name__ == '__main__':
    args = parse_args()
    args.config_file = 'SegmenTron/configs/cityscapes_fast_scnn.yaml'
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    evaluator = Pretreatment(args)
    evaluator.process()
