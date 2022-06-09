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

from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import torch

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            if args.distributed == 1:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(MyConcatDataset(datasets))
                self.loader_train = torch.utils.data.DataLoader(
                    MyConcatDataset(datasets),
                    batch_size=args.batch_size,
                    shuffle=(self.train_sampler is None),
                    num_workers=args.n_threads,
                    pin_memory=False,
                    sampler=self.train_sampler,
                    drop_last=True
                )
            else:
                self.train_sampler = None
                self.loader_train = dataloader.DataLoader(
                    MyConcatDataset(datasets),
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=False,
                    num_workers=args.n_threads,
                )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=False,
                    num_workers=args.n_threads,
                )
            )
