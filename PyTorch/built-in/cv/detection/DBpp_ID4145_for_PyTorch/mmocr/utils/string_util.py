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
class StringStrip:
    """Removing the leading and/or the trailing characters based on the string
    argument passed.

    Args:
        strip (bool): Whether remove characters from both left and right of
            the string. Default: True.
        strip_pos (str): Which position for removing, can be one of
            ('both', 'left', 'right'), Default: 'both'.
        strip_str (str|None): A string specifying the set of characters
            to be removed from the left and right part of the string.
            If None, all leading and trailing whitespaces
            are removed from the string. Default: None.
    """

    def __init__(self, strip=True, strip_pos='both', strip_str=None):
        assert isinstance(strip, bool)
        assert strip_pos in ('both', 'left', 'right')
        assert strip_str is None or isinstance(strip_str, str)

        self.strip = strip
        self.strip_pos = strip_pos
        self.strip_str = strip_str

    def __call__(self, in_str):

        if not self.strip:
            return in_str

        if self.strip_pos == 'left':
            return in_str.lstrip(self.strip_str)
        elif self.strip_pos == 'right':
            return in_str.rstrip(self.strip_str)
        else:
            return in_str.strip(self.strip_str)
