# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# -*- coding:utf-8 -*-
from .log_uti import mlog

from ..mod_modify.interface import BaseNode


def check_node_fix_shape(node: BaseNode, expect_len, fix_idx, fix_value: int):
    shape = node.shape
    if shape is None:
        mlog("node:{} shape is None".format(node.name))
        return False
    valid_param = (len(shape) == expect_len and expect_len > 0 and
                   fix_idx < expect_len and fix_idx >= -expect_len)
    if valid_param:
        return shape[fix_idx] == fix_value
    else:
        mlog("params not match. shape:{}, expect_len:{}, fix_idx:{}".format(
            shape, expect_len, fix_idx))
        return False
