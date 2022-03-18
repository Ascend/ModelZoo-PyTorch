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

from torch.utils import data
from .pascal_voc import VOCSegmentation

datasets = {
    'pascal_voc': VOCSegmentation
}

def get_num_classes(args):
    return datasets[args.dataset.lower()].NUM_CLASS

def get_class_names(args):
    return datasets[args.dataset.lower()].CLASSES

def get_dataloader(args, cfg, split, batch_size=None, test_mode=None):
    assert split in ('train', 'train_voc', 'val'), "Unknown split '{}'".format(split)

    dataset_name = args.dataset.lower()
    dataset_cls = datasets[dataset_name]
    dataset = dataset_cls(cfg, split, test_mode)

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    shuffle, drop_last = [True, True] if split == 'train' else [False, False]

    if batch_size is None:
        batch_size = cfg.TRAIN.BATCH_SIZE

    return data.DataLoader(dataset, batch_size=batch_size,
                           drop_last=drop_last, shuffle=shuffle, 
                           **kwargs)
