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

from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathHandler
from iopath.common.file_io import PathManager as PathManagerBase

__all__ = ["PathManager", "PathHandler"]


PathManager = PathManagerBase()
"""
This is a detectron2 project-specific PathManager.
We try to stay away from global PathManager in fvcore as it
introduces potential conflicts among other libraries.
"""


class Detectron2Handler(PathHandler):
    """
    Resolve anything that's hosted under detectron2's namespace.
    """

    PREFIX = "detectron2://"
    S3_DETECTRON2_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path, **kwargs):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.S3_DETECTRON2_PREFIX + name, **kwargs)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())
PathManager.register_handler(Detectron2Handler())
