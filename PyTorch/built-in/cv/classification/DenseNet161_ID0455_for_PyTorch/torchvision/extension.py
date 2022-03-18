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
#_C = None


def _lazy_import():
    """
    Make sure that CUDA versions match between the pytorch install and torchvision install
    """
    global _C
    if _C is not None:
        return _C
    import torch
    from torchvision import _C as C
    _C = C
    if hasattr(_C, "CUDA_VERSION") and torch.version.cuda is not None:
        tv_version = str(_C.CUDA_VERSION)
        if int(tv_version) < 10000:
            tv_major = int(tv_version[0])
            tv_minor = int(tv_version[2])
        else:
            tv_major = int(tv_version[0:2])
            tv_minor = int(tv_version[3])
        t_version = torch.version.cuda
        t_version = t_version.split('.')
        t_major = int(t_version[0])
        t_minor = int(t_version[1])
        if t_major != tv_major or t_minor != tv_minor:
            raise RuntimeError("Detected that PyTorch and torchvision were compiled with different CUDA versions. "
                               "PyTorch has CUDA Version={}.{} and torchvision has CUDA Version={}.{}. "
                               "Please reinstall the torchvision that matches your PyTorch install."
                               .format(t_major, t_minor, tv_major, tv_minor))
    return _C
