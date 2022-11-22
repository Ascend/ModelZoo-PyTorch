# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright 2020 Huawei Technologies Co., Ltd
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
# --------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.dev/open-mmlab/mmcv
import os.path as osp
import sys
from importlib import import_module


def load_config(cfg_dir: str) -> dict:
    assert cfg_dir.endswith('.py')
    root_path, file_name = osp.split(cfg_dir)
    temp_module = osp.splitext(file_name)[0]
    sys.path.insert(0, root_path)
    mod = import_module(temp_module)
    sys.path.pop(0)
    cfg_dict = {
        k: v
        for k, v in mod.__dict__.items() if not k.startswith('__')
    }
    del sys.modules[temp_module]
    return cfg_dict
