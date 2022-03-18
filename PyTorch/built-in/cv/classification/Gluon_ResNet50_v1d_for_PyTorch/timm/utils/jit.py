#     Copyright [yyyy] [name of copyright owner]
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

""" JIT scripting/tracing utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch


def set_jit_legacy():
    """ Set JIT executor to legacy w/ support for op fusion
    This is hopefully a temporary need in 1.5/1.5.1/1.6 to restore performance due to changes
    in the JIT exectutor. These API are not supported so could change.
    """
    #
    assert hasattr(torch._C, '_jit_set_profiling_executor'), "Old JIT behavior doesn't exist!"
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_gpu(True)
    #torch._C._jit_set_texpr_fuser_enabled(True)
