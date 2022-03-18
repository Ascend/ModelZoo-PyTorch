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
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools

from fairseq.hub_utils import BPEHubInterface as bpe  # noqa
from fairseq.hub_utils import TokenizerHubInterface as tokenizer  # noqa
from fairseq.models import MODEL_REGISTRY


dependencies = [
    'numpy',
    'regex',
    'requests',
    'torch',
]


# torch.hub doesn't build Cython components, so if they are not found then try
# to build them here
try:
    import fairseq.data.token_block_utils_fast
except (ImportError, ModuleNotFoundError):
    try:
        import cython
        import os
        from setuptools import sandbox
        sandbox.run_setup(
            os.path.join(os.path.dirname(__file__), 'setup.py'),
            ['build_ext', '--inplace'],
        )
    except (ImportError, ModuleNotFoundError):
        print(
            'Unable to build Cython components. Please make sure Cython is '
            'installed if the torch.hub model you are loading depends on it.'
        )


for _model_type, _cls in MODEL_REGISTRY.items():
    for model_name in _cls.hub_models().keys():
        globals()[model_name] = functools.partial(
            _cls.from_pretrained,
            model_name,
        )
    # to simplify the interface we only expose named models
    # globals()[_model_type] = _cls.from_pretrained
