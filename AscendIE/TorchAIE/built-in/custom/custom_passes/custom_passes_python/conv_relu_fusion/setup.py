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

import os
from setuptools import setup, Extension
from torch.utils import cpp_extension

# library_dirs should point to the libtorch_aie.so and the libascendie.so
# include_dirs should point to the dir that include the headers
TORCH_AIE_PATH = "/opt/buildtools/Python-3.9.11/lib/python3.9/site-packages/torch_aie"
ASCEND_IE_PATH = "/usr/local/Ascend/aie/latest"

ext_modules = [
    cpp_extension.CppExtension('custom_passes_reduce', ['./csrc/conv_relu_fusion.cpp'],
                                library_dirs=[(TORCH_AIE_PATH + "/lib/")],
                                libraries=["torch_aie"],
                                include_dirs=[
                                    TORCH_AIE_PATH + "/include/",
                                    ASCEND_IE_PATH + "/include/"
                                ])
]

setup(
    name='custom_passes_reduce',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
