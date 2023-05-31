# Copyright 2021 Huawei Technologies Co., Ltd
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

# coding=utf-8

import setuptools

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='tensorrt_dynamic',
    version='0.0.1',
    description='TensorRT dynamic infer tool',
    long_description=long_description,
    url='TensorRT url',
    packages=setuptools.find_packages(),
    keywords='TensorRT tool',
    install_requires=required,
    python_requires='>=3.7'
)