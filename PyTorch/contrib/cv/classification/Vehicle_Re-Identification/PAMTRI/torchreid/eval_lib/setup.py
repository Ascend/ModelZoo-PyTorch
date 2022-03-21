# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License-NC
# See LICENSE.txt for details
#
# Author: Zheng Tang (tangzhengthomas@gmail.com)
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()
print(numpy_include)

ext_modules = [Extension("cython_eval",
                         ["eval.pyx"],
                         libraries=["m"],
                         include_dirs=[numpy_include],
                         extra_compile_args=["-ffast-math", "-Wno-cpp", "-Wno-unused-function"]
                         ),
               ]

setup(
    name='eval_lib',
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules)
