
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

# Copyright (c) Open-MMLab. All rights reserved.    
# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmdet.utils import get_caller_name, log_img_scale


def callee_func():
    caller_name = get_caller_name()
    return caller_name


class CallerClassForTest:

    def __init__(self):
        self.caller_name = callee_func()


def test_get_caller_name():
    # test the case that caller is a function
    caller_name = callee_func()
    assert caller_name == 'test_get_caller_name'

    # test the case that caller is a method in a class
    caller_class = CallerClassForTest()
    assert caller_class.caller_name == 'CallerClassForTest.__init__'


def test_log_img_scale():
    img_scale = (800, 1333)
    done_logging = log_img_scale(img_scale)
    assert done_logging

    img_scale = (1333, 800)
    done_logging = log_img_scale(img_scale, shape_order='wh')
    assert done_logging

    with pytest.raises(ValueError):
        img_scale = (1333, 800)
        done_logging = log_img_scale(img_scale, shape_order='xywh')

    img_scale = (640, 640)
    done_logging = log_img_scale(img_scale, skip_square=False)
    assert done_logging

    img_scale = (640, 640)
    done_logging = log_img_scale(img_scale, skip_square=True)
    assert not done_logging
