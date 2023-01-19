# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from .builder import RUNNER_BUILDERS, RUNNERS


@RUNNER_BUILDERS.register_module()
class DefaultRunnerConstructor:
    """Default constructor for runners.

    Custom existing `Runner` like `EpocBasedRunner` though `RunnerConstructor`.
    For example, We can inject some new properties and functions for `Runner`.

    Example:
        >>> from mmcv.runner import RUNNER_BUILDERS, build_runner
        >>> # Define a new RunnerReconstructor
        >>> @RUNNER_BUILDERS.register_module()
        >>> class MyRunnerConstructor:
        ...     def __init__(self, runner_cfg, default_args=None):
        ...         if not isinstance(runner_cfg, dict):
        ...             raise TypeError('runner_cfg should be a dict',
        ...                             f'but got {type(runner_cfg)}')
        ...         self.runner_cfg = runner_cfg
        ...         self.default_args = default_args
        ...
        ...     def __call__(self):
        ...         runner = RUNNERS.build(self.runner_cfg,
        ...                                default_args=self.default_args)
        ...         # Add new properties for existing runner
        ...         runner.my_name = 'my_runner'
        ...         runner.my_function = lambda self: print(self.my_name)
        ...         ...
        >>> # build your runner
        >>> runner_cfg = dict(type='EpochBasedRunner', max_epochs=40,
        ...                   constructor='MyRunnerConstructor')
        >>> runner = build_runner(runner_cfg)
    """

    def __init__(self, runner_cfg: dict, default_args: Optional[dict] = None):
        if not isinstance(runner_cfg, dict):
            raise TypeError('runner_cfg should be a dict',
                            f'but got {type(runner_cfg)}')
        self.runner_cfg = runner_cfg
        self.default_args = default_args

    def __call__(self):
        return RUNNERS.build(self.runner_cfg, default_args=self.default_args)
