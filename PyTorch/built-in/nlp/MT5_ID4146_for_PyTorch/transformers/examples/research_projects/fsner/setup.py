# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import setuptools

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURRENT_PATH, '../../../../url.ini'), 'r') as _f:
    content = _f.read()
    fsner_url = content.split('fsner_url=')[1].split('\n')[0]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fsner",
    version="0.0.1",
    author="msi sayef",
    author_email="msi.sayef@gmail.com",
    description="Few-shot Named Entity Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=fsner_url,
    project_urls={
        "Bug Tracker": "https://github.com/huggingface/transformers/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=["torch>=1.9.0", "transformers>=4.9.2"],
)
