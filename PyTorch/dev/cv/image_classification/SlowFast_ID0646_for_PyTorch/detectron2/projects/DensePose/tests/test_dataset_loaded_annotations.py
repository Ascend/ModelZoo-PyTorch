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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from densepose.data.datasets.builtin import COCO_DATASETS, DENSEPOSE_ANNOTATIONS_DIR, LVIS_DATASETS
from densepose.data.datasets.coco import load_coco_json
from densepose.data.datasets.lvis import load_lvis_json
from densepose.data.utils import maybe_prepend_base_path
from densepose.structures import DensePoseDataRelative


class TestDatasetLoadedAnnotations(unittest.TestCase):
    COCO_DATASET_DATA = {
        "densepose_coco_2014_train": {"n_instances": 39210},
        "densepose_coco_2014_minival": {"n_instances": 2243},
        "densepose_coco_2014_minival_100": {"n_instances": 164},
        "densepose_coco_2014_valminusminival": {"n_instances": 7297},
        "densepose_coco_2014_train_cse": {"n_instances": 39210},
        "densepose_coco_2014_minival_cse": {"n_instances": 2243},
        "densepose_coco_2014_minival_100_cse": {"n_instances": 164},
        "densepose_coco_2014_valminusminival_cse": {"n_instances": 7297},
        "densepose_chimps": {"n_instances": 930},
        "posetrack2017_train": {"n_instances": 8274},
        "posetrack2017_val": {"n_instances": 4753},
        "lvis_v05_train": {"n_instances": 5186},
        "lvis_v05_val": {"n_instances": 1037},
    }

    LVIS_DATASET_DATA = {
        "densepose_lvis_v1_train1": {"n_instances": 3394},
        "densepose_lvis_v1_train2": {"n_instances": 1800},
        "densepose_lvis_v1_val": {"n_instances": 1037},
        "densepose_lvis_v1_val_animals_100": {"n_instances": 89},
    }

    def generic_coco_test(self, dataset_info):
        if dataset_info.name not in self.COCO_DATASET_DATA:
            return
        n_inst = self.COCO_DATASET_DATA[dataset_info.name]["n_instances"]
        self.generic_test(dataset_info, n_inst, load_coco_json)

    def generic_lvis_test(self, dataset_info):
        if dataset_info.name not in self.LVIS_DATASET_DATA:
            return
        n_inst = self.LVIS_DATASET_DATA[dataset_info.name]["n_instances"]
        self.generic_test(dataset_info, n_inst, load_lvis_json)

    def generic_test(self, dataset_info, n_inst, loader_fun):
        datasets_root = DENSEPOSE_ANNOTATIONS_DIR
        annotations_fpath = maybe_prepend_base_path(datasets_root, dataset_info.annotations_fpath)
        images_root = maybe_prepend_base_path(datasets_root, dataset_info.images_root)
        image_annotation_dicts = loader_fun(
            annotations_json_file=annotations_fpath,
            image_root=images_root,
            dataset_name=dataset_info.name,
        )
        num_valid = sum(
            1
            for image_annotation_dict in image_annotation_dicts
            for ann in image_annotation_dict["annotations"]
            if DensePoseDataRelative.validate_annotation(ann)[0]
        )
        self.assertEqual(num_valid, n_inst)


def coco_test_fun(dataset_info):
    return lambda self: self.generic_coco_test(dataset_info)


for dataset_info in COCO_DATASETS:
    setattr(
        TestDatasetLoadedAnnotations,
        f"test_coco_builtin_loaded_annotations_{dataset_info.name}",
        coco_test_fun(dataset_info),
    )


def lvis_test_fun(dataset_info):
    return lambda self: self.generic_lvis_test(dataset_info)


for dataset_info in LVIS_DATASETS:
    setattr(
        TestDatasetLoadedAnnotations,
        f"test_lvis_builtin_loaded_annotations_{dataset_info.name}",
        lvis_test_fun(dataset_info),
    )
