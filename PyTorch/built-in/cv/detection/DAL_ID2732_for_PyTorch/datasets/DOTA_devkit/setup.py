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

"""
    setup.py file for SWIG example
"""
from distutils.core import setup, Extension
import numpy

polyiou_module = Extension('_polyiou',
                           sources=['polyiou_wrap.cxx', 'polyiou.cpp'],
                           )
setup(name = 'polyiou',
      version = '0.1',
      author = "SWIG Docs",
      description = """Simple swig example from docs""",
      ext_modules = [polyiou_module],
      py_modules = ["polyiou"],
)
