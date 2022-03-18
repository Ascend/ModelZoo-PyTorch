# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import json
import yaml
from pathlib import Path
import os


def make_parent(file_path):
    file_path = Path(file_path)
    os.makedirs(file_path.parent, exist_ok=True)


def pickle_dump(python_object, file_path):
    make_parent(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(python_object, f)


def pickle_load(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def json_load(file_path):
    with open(file_path, 'r') as fp:
        return json.load(fp)


def yaml_dump(python_object, file_path):
    make_parent(file_path)
    with open(file_path, 'w') as f:
        yaml.dump(python_object, f, default_flow_style=False)


def yaml_load(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f)
