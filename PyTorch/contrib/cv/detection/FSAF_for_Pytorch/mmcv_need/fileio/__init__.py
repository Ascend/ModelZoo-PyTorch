# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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

# Copyright (c) Open-MMLab. All rights reserved.
from .file_client import BaseStorageBackend, FileClient
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler
from .io import dump, load, register_handler
from .parse import dict_from_file, list_from_file

__all__ = [
    'BaseStorageBackend', 'FileClient', 'load', 'dump', 'register_handler',
    'BaseFileHandler', 'JsonHandler', 'PickleHandler', 'YamlHandler',
    'list_from_file', 'dict_from_file'
]
