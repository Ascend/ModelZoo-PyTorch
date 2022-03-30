# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
#
#
# Copyright (c) Facebook, Inc. and its affiliates.
import importlib
from pathlib import Path

_PROJECTS = {
    "point_rend": "PointRend",
    "deeplab": "DeepLab",
    "panoptic_deeplab": "Panoptic-DeepLab",
}
_PROJECT_ROOT = Path(__file__).parent.parent.parent / "projects"

if _PROJECT_ROOT.is_dir():
    # This is true only for in-place installation (pip install -e, setup.py develop),
    # where setup(package_dir=) does not work: https://github.com/pypa/setuptools/issues/230

    class _D2ProjectsFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if not name.startswith("detectron2.projects."):
                return
            project_name = name.split(".")[-1]
            project_dir = _PROJECTS.get(project_name)
            if not project_dir:
                return
            target_file = _PROJECT_ROOT / f"{project_dir}/{project_name}/__init__.py"
            if not target_file.is_file():
                return
            return importlib.util.spec_from_file_location(name, target_file)

    import sys

    sys.meta_path.append(_D2ProjectsFinder())
