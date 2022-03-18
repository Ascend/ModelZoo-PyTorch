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


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        name='toolkit.utils.region',
        sources=[
            'toolkit/utils/region.pyx',
            'toolkit/utils/src/region.c',
        ],
        include_dirs=[
            'toolkit/utils/src'
        ]
    )
]

setup(
    name='toolkit',
    packages=['toolkit'],
    ext_modules=cythonize(ext_modules)
)
