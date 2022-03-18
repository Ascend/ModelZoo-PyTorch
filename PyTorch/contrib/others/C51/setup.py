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

from setuptools import setup, find_packages
import sys

print('Please install OpenAI Baselines (commit 8e56dd) and requirement.txt')
if not (sys.version.startswith('3.5') or sys.version.startswith('3.6')):
    raise Exception('Only Python 3.5 and 3.6 are supported')

setup(name='deep_rl',
      packages=[package for package in find_packages()
                if package.startswith('deep_rl')],
      install_requires=[],
      description="Modularized Implementation of Deep RL Algorithms",
      author="Shangtong Zhang",
      url='https://github.com/ShangtongZhang/DeepRL',
      author_email="zhangshangtong.cpp@gmail.com",
      version="1.5")