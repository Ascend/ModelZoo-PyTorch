# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================

from setuptools import find_packages, setup

setup(
    name="slowfast",
    version="1.0",
    author="FAIR",
    url="unknown",
    description="SlowFast Video Understanding",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "av",
        "matplotlib",
        "termcolor>=1.1",
        "simplejson",
        "tqdm",
        "psutil",
        "matplotlib",
        "detectron2",
        "opencv-python",
        "pandas",
        "torchvision",
        "Pillow",
        "sklearn",
        "tensorboard",
    ],
    extras_require={"tensorboard_video_visualization": ["moviepy"]},
    packages=find_packages(exclude=("configs", "tests")),
)
