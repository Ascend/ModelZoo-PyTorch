#     Copyright 2021 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

#!/usr/bin/env python3
import os
import setuptools
from distutils.core import setup

with open("README.md") as f:
    long_description = f.read()

with open(os.path.join("speechbrain", "version.txt")) as f:
    version = f.read().strip()

setup(
    name="speechbrain",
    version=version,
    description="All-in-one speech toolkit in pure Python and Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mirco Ravanelli & Others",
    author_email="speechbrain@gmail.com",
    packages=setuptools.find_packages(),
    package_data={"speechbrain": ["version.txt", "log-config.yaml"]},
    install_requires=[
        "hyperpyyaml",
        "joblib",
        "numpy",
        "packaging",
	    "scipy",
        "sentencepiece",
		"torch",
		"torchaudio",
        "tqdm",
        "huggingface_hub",
    ],
    python_requires=">=3.7",
    url="https://speechbrain.github.io/",
)
