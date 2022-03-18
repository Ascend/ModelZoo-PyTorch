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

from typing import Optional

from detectron2.data import DatasetCatalog, MetadataCatalog

from ..utils import maybe_prepend_base_path
from .dataset_type import DatasetType

CHIMPNSEE_DATASET_NAME = "chimpnsee"


def register_dataset(datasets_root: Optional[str] = None):
    def empty_load_callback():
        pass

    video_list_fpath = maybe_prepend_base_path(
        datasets_root,
        "chimpnsee/cdna.eva.mpg.de/video_list.txt",
    )
    video_base_path = maybe_prepend_base_path(datasets_root, "chimpnsee/cdna.eva.mpg.de")

    DatasetCatalog.register(CHIMPNSEE_DATASET_NAME, empty_load_callback)
    MetadataCatalog.get(CHIMPNSEE_DATASET_NAME).set(
        dataset_type=DatasetType.VIDEO_LIST,
        video_list_fpath=video_list_fpath,
        video_base_path=video_base_path,
        category="chimpanzee",
    )
