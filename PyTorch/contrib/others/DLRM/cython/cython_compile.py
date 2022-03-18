# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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
#
# Description: compile .so from python code


from __future__ import absolute_import, division, print_function, unicode_literals

from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [
    Extension(
        "data_utils_cython",
        ["data_utils_cython.pyx"],
        extra_compile_args=['-O3'],
        extra_link_args=['-O3'],
    )
]

setup(
    name='data_utils_cython',
    ext_modules=cythonize(ext_modules)
)
