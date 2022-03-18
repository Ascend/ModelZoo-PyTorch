#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
import re
from pathlib import Path
from setuptools import find_packages, setup

try:
    import torch  # noqa: F401
except ImportError:
    raise Exception(
        """
You must install PyTorch prior to installing DensePose:
pip install torch

For more information:
    https://pytorch.org/get-started/locally/
    """
    )


def get_detectron2_current_version():
    """Version is not available for import through Python since it is
    above the top level of the package. Instead, we parse it from the
    file with a regex."""
    # Get version info from detectron2 __init__.py
    version_source = (Path(__file__).parents[2] / "detectron2" / "__init__.py").read_text()
    version_number = re.findall(r'__version__ = "([0-9\.]+)"', version_source)[0]
    return version_number


setup(
    name="detectron2-densepose",
    author="FAIR",
    version=get_detectron2_current_version(),
    url="https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "av>=8.0.3",
        "detectron2@git+https://github.com/facebookresearch/detectron2.git",
        "opencv-python-headless>=4.5.3.56",
        "scipy>=1.5.4",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
    ],
)
