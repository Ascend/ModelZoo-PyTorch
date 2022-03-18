# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import importlib
import os
import pkgutil
from collections import namedtuple

import torch

if torch.__version__ != 'parrots':

    def load_ext(name, funcs):
        ext = importlib.import_module('mmcv.' + name)
        for fun in funcs:
            assert hasattr(ext, fun), f'{fun} miss in module {name}'
        return ext
else:
    from parrots import extension

    has_return_value_ops = [
        'nms', 'softnms', 'nms_match', 'nms_rotated', 'top_pool_forward',
        'top_pool_backward', 'bottom_pool_forward', 'bottom_pool_backward',
        'left_pool_forward', 'left_pool_backward', 'right_pool_forward',
        'right_pool_backward', 'fused_bias_leakyrelu', 'upfirdn2d'
    ]

    def load_ext(name, funcs):
        ExtModule = namedtuple('ExtModule', funcs)
        ext_list = []
        lib_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        for fun in funcs:
            if fun in has_return_value_ops:
                ext_list.append(extension.load(fun, name, lib_dir=lib_root).op)
            else:
                ext_list.append(
                    extension.load(fun, name, lib_dir=lib_root).op_)
        return ExtModule(*ext_list)


def check_ops_exist():
    ext_loader = pkgutil.find_loader('mmcv._ext')
    return ext_loader is not None
