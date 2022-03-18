# Copyright 2020 Huawei Technologies Co., Ltd## Licensed under the Apache License, Version 2.0 (the "License");# you may not use this file except in compliance with the License.# You may obtain a copy of the License at## http://www.apache.org/licenses/LICENSE-2.0## Unless required by applicable law or agreed to in writing, software# distributed under the License is distributed on an "AS IS" BASIS,# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.# See the License for the specific language governing permissions and# limitations under the License.# ============================================================================from mmcv.utils import collect_env as collect_basic_env
from mmcv.utils import get_git_hash

import mmaction


def collect_env():
    env_info = collect_basic_env()
    env_info['MMAction2'] = (
        mmaction.__version__ + '+' + '9')
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
