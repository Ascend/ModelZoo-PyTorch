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
import json
import warnings

from mmocr.datasets.builder import PARSERS
from mmocr.utils import StringStrip


@PARSERS.register_module()
class LineStrParser:
    """Parse string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in result dict.
        keys_idx (list[int]): Value index in sub-string list
            for each key above.
        separator (str): Separator to separate string to list of sub-string.
    """

    def __init__(self,
                 keys=['filename', 'text'],
                 keys_idx=[0, 1],
                 separator=' ',
                 **kwargs):
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator
        self.strip_cls = StringStrip(**kwargs)

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        line_str = data_ret[map_index]
        line_str = self.strip_cls(line_str)
        if len(line_str.split(' ')) > 2:
            msg = 'More than two blank spaces were detected. '
            msg += 'Please use LineJsonParser to handle '
            msg += 'annotations with blanks. '
            msg += 'Check Doc '
            msg += 'https://mmocr.readthedocs.io/en/latest/'
            msg += 'tutorials/blank_recog.html '
            msg += 'for details.'
            warnings.warn(msg)
        line_str = line_str.split(self.separator)
        if len(line_str) <= max(self.keys_idx):
            raise Exception(
                f'key index: {max(self.keys_idx)} out of range: {line_str}')

        line_info = {}
        for i, key in enumerate(self.keys):
            line_info[key] = line_str[self.keys_idx[i]]
        return line_info


@PARSERS.register_module()
class LineJsonParser:
    """Parse json-string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in both json-string and result dict.
    """

    def __init__(self, keys=[]):
        assert isinstance(keys, list)
        assert len(keys) > 0
        self.keys = keys

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        json_str = data_ret[map_index]
        line_json_obj = json.loads(json_str)
        line_info = {}
        for key in self.keys:
            if key not in line_json_obj:
                raise Exception(f'key {key} not in line json {line_json_obj}')
            line_info[key] = line_json_obj[key]

        return line_info
