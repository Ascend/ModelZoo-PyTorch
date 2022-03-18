# Copyright 2021 Huawei Technologies Co., Ltd
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

#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='xlm',
    version='1.0',
    description='Text simplification',
    author='Guillaume Lample, Alexis Conneau',
    author_email='glample@fb.com, aconneau@fb.com',
    packages=find_packages(),
)
