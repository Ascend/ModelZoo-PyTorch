#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
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
#
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import pickle
import sys
import unittest
from functools import partial
import torch
from iopath.common.file_io import LazyPath

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.data import (
    DatasetFromList,
    MapDataset,
    ToIterableDataset,
    build_batch_data_loader,
    build_detection_train_loader,
)
from detectron2.data.samplers import InferenceSampler, TrainingSampler


def _a_slow_func(x):
    return "path/{}".format(x)


class TestDatasetFromList(unittest.TestCase):
    # Failing for py3.6, likely due to pickle
    @unittest.skipIf(sys.version_info.minor <= 6, "Not supported in Python 3.6")
    def test_using_lazy_path(self):
        dataset = []
        for i in range(10):
            dataset.append({"file_name": LazyPath(partial(_a_slow_func, i))})

        dataset = DatasetFromList(dataset)
        for i in range(10):
            path = dataset[i]["file_name"]
            self.assertTrue(isinstance(path, LazyPath))
            self.assertEqual(os.fspath(path), _a_slow_func(i))


class TestMapDataset(unittest.TestCase):
    @staticmethod
    def map_func(x):
        if x == 2:
            return None
        return x * 2

    def test_map_style(self):
        ds = DatasetFromList([1, 2, 3])
        ds = MapDataset(ds, TestMapDataset.map_func)
        self.assertEqual(ds[0], 2)
        self.assertEqual(ds[2], 6)
        self.assertIn(ds[1], [2, 6])

    def test_iter_style(self):
        class DS(torch.utils.data.IterableDataset):
            def __iter__(self):
                yield from [1, 2, 3]

        ds = DS()
        ds = MapDataset(ds, TestMapDataset.map_func)
        self.assertIsInstance(ds, torch.utils.data.IterableDataset)

        data = list(iter(ds))
        self.assertEqual(data, [2, 6])

    def test_pickleability(self):
        ds = DatasetFromList([1, 2, 3])
        ds = MapDataset(ds, lambda x: x * 2)
        ds = pickle.loads(pickle.dumps(ds))
        self.assertEqual(ds[0], 2)


@unittest.skipIf(os.environ.get("CI"), "Skipped OSS testing due to COCO data requirement.")
class TestDataLoader(unittest.TestCase):
    def _get_kwargs(self):
        # get kwargs of build_detection_train_loader
        cfg = model_zoo.get_config("common/data/coco.py").dataloader.train
        cfg.dataset.names = "coco_2017_val_100"
        cfg.pop("_target_")
        kwargs = {k: instantiate(v) for k, v in cfg.items()}
        return kwargs

    def test_build_dataloader(self):
        kwargs = self._get_kwargs()
        dl = build_detection_train_loader(**kwargs)
        next(iter(dl))

    def test_build_iterable_dataloader(self):
        kwargs = self._get_kwargs()
        ds = DatasetFromList(kwargs.pop("dataset"))
        ds = ToIterableDataset(ds, TrainingSampler(len(ds)))
        dl = build_detection_train_loader(dataset=ds, **kwargs)
        next(iter(dl))

    def test_build_dataloader_inference(self):
        N = 96
        ds = DatasetFromList(list(range(N)))
        sampler = InferenceSampler(len(ds))
        dl = build_batch_data_loader(ds, sampler, 8, num_workers=3)

        data = list(iter(dl))
        data = [x for batch in data for x in batch]  # flatten the batches
        self.assertEqual(len(data), N)
        self.assertEqual(set(data), set(range(N)))
