#!/usr/bin/env python3
# ============================================================================
# Copyright 2022 Huawei Technologies Co., Ltd
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

from setuptools import find_packages, setup
setup(
    name="markuplmft",
    version="0.1",
    author="MarkupLM Team",
    packages=find_packages(),
    python_requires=">=3.7",
    extras_require={"dev": ["flake8", "isort", "black"]},
)