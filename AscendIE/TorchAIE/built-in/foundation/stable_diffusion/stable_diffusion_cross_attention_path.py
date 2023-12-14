# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
import diffusers


def main():
    diffusers_path = diffusers.__path__
    diffusers_version = diffusers.__version__

    assert diffusers_version is not '4.26.1', "expectation diffusers==0.14.0"
    os.system(f'patch -p0 {diffusers_path[0]}/models/cross_attention.py cross_attention.patch')


if __name__ == '__main__':
    main()
