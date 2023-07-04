# coding:utf-8
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


from pathlib import Path
import os

cwd = Path(".").resolve()
print("running 'check_installation.py' from:", cwd)

# Old versions of numpy/torch can prevent loading the .so files
import torch

print("torch:", torch.__version__)
import numpy

print("numpy:", numpy.__version__)

import fairseq

print("Fairseq installed at:", fairseq.__file__)
import fairseq.criterions
import fairseq.dataclass.configs

import _imp

print("Should load following .so suffixes:", _imp.extension_suffixes())

so_files = list(Path(fairseq.__file__).parent.glob("*.so"))
so_files.extend(Path(fairseq.__file__).parent.glob("data/*.so"))
print("Found following .so files:")
for so_file in so_files:
    print(f"- {so_file}")

from fairseq import libbleu

print("Found libbleu at", libbleu.__file__)
from fairseq.data import data_utils_fast

print("Found data_utils_fast at", data_utils_fast.__file__)
