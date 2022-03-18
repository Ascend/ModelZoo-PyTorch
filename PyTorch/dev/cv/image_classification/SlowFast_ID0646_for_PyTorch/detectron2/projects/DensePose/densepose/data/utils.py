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
from typing import Dict, Optional

from detectron2.config import CfgNode


def is_relative_local_path(path: str):
    path_str = os.fsdecode(path)
    return ("://" not in path_str) and not os.path.isabs(path)


def maybe_prepend_base_path(base_path: Optional[str], path: str):
    """
    Prepends the provided path with a base path prefix if:
    1) base path is not None;
    2) path is a local path
    """
    if base_path is None:
        return path
    if is_relative_local_path(path):
        return os.path.join(base_path, path)
    return path


def get_class_to_mesh_name_mapping(cfg: CfgNode) -> Dict[int, str]:
    return {
        int(class_id): mesh_name
        for class_id, mesh_name in cfg.DATASETS.CLASS_TO_MESH_NAME_MAPPING.items()
    }


def get_category_to_class_mapping(dataset_cfg: CfgNode) -> Dict[str, int]:
    return {
        category: int(class_id)
        for category, class_id in dataset_cfg.CATEGORY_TO_CLASS_MAPPING.items()
    }
