
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

# This file is used for package installation and find default exp file

import importlib
import sys
from pathlib import Path

_EXP_PATH = Path(__file__).resolve().parent.parent.parent.parent / "exps" / "default"

if _EXP_PATH.is_dir():
    # This is true only for in-place installation (pip install -e, setup.py develop),
    # where setup(package_dir=) does not work: https://github.com/pypa/setuptools/issues/230

    class _ExpFinder(importlib.abc.MetaPathFinder):
        
        def find_spec(self, name, path, target=None):
            if not name.startswith("yolox.exp.default"):
                return
            project_name = name.split(".")[-1] + ".py"
            target_file = _EXP_PATH / project_name
            if not target_file.is_file():
                return
            return importlib.util.spec_from_file_location(name, target_file)

    sys.meta_path.append(_ExpFinder())
