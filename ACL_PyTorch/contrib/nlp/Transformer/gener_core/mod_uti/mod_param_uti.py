# -*- coding:utf-8 -*-
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
import os
from .log_uti import mlog
from enum import Enum


class ModFMK(Enum):
    CAFFE = 0
    TF = 3
    ONNX = 5


def get_third_party_mod_info(mod_path: str) -> tuple:
    mod_infos = mod_path.split(",")
    if (len(mod_infos) == 1):  # pb or onnx
        _, mod_name = os.path.split(mod_infos[0])
        valid_ext = ((mod_name.endswith(".pb") or mod_name.endswith(".onnx")))
        valid_path = os.path.exists(mod_infos[0])
        mod_fmk = ModFMK.TF if mod_name.endswith(".pb") else ModFMK.ONNX
    elif (len(mod_infos) == 2):  # caffe
        _, mod_name = os.path.split(mod_infos[0])
        _, weight_name = os.path.split(mod_infos[1])
        valid_path = (os.path.exists(mod_infos[0]) and
                      os.path.exists(mod_infos[1]))
        valid_ext = (mod_name.endswith(".prototxt") and
                     weight_name.endswith(".caffemodel"))
        mod_fmk = ModFMK.CAFFE
    else:
        mlog("unknown mod_path:{}, sep cnt:{} not 1 or 2".format(
            mod_path, len(mod_infos)))
        return None, None

    if (not valid_path):
        mlog("mod_path:{} not exist".format(mod_path))
        return None, None
    if (not valid_ext):
        mlog("mod_path:{} not pb/onnx/caffemodel".format(mod_path))
        return None, None
    return mod_infos, mod_fmk


def check_params_dtype_len(params: dict, check_items: dict):
    valid_input = isinstance(params, dict) and isinstance(check_items, dict)
    if (not valid_input):
        mlog("invalid input. params:{}, check_items:{}".format(
            params, check_items))
        return False

    for key, item in check_items.items():
        if (not isinstance(item, dict)):
            mlog("check item need to be dict. {}:{}".format(key, item))
        if ("dtype" not in item):
            mlog("check item need to have dtype. {}:{}".format(key, item))

        dtype = item.get("dtype")
        min_v = item.get("min")
        max_v = item.get("max")

        if (key not in params):
            mlog("params not have key:{}".format(key))
            return False
        param_value = params.get(key)
        if (not isinstance(param_value, dtype)):
            mlog("param {}:{} is not {}".format(key, param_value, dtype))
            return False

        valid_param = True
        if (isinstance(param_value, (int, float))):
            valid_param = _check_value(param_value, min_v, max_v)
        elif (isinstance(param_value, (str, tuple, list))):
            valid_param = _check_len(param_value, min_v, max_v)
        else:
            pass    # skip check for unknown param dtype

        if (not valid_param):
            mlog("{}:{} invalid. min len:{}, max len:{}".format(
                key, param_value, min_v, max_v))
            return False
    return True


def check_pos_values(name, value, value_types=(int, float)):
    if (isinstance(value, value_types)):
        if (value <= 0):
            mlog("{} val {} isn't positive".format(name, value))
    elif (isinstance(value, (list, tuple))):
        if (not _check_list_tuple_elements(value, 0, None)):
            mlog("{} val {} isn't positive.".format(name, value))
    else:
        mlog("{} unsupported value type. {}".format(name, type(value)))
    return True


def _check_value(tar_v, min_v, max_v):
    if (min_v is not None and tar_v <= min_v):
        return False

    if (max_v is not None and tar_v >= max_v):
        return False

    return True


def _check_len(tar_p, min_len, max_len):
    tar_len = len(tar_p)
    return _check_value(tar_len, min_len, max_len)


def _check_list_tuple_elements(tar, min_v, max_v):
    for cur_v in tar:
        if (not _check_value(cur_v, min_v, max_v)):
            return False
    return True
