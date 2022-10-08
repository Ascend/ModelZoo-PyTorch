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
import logging
import subprocess as subp

from .log_uti import mlog
from .mod_param_uti import ModFMK


def check_support(confs: dict, support: dict):
    if not isinstance(confs, dict):
        mlog("confs:{} is not dict.".format(confs))
        return False
    for key, value in support.items():
        if key not in confs:
            mlog("key:{} not in confs:{}".format(key, confs))
            return False

        if confs[key] not in value:
            mlog("confs[{}]:{} not in support list:{}".format(
                key, confs[key], value))
            return False
    return True


def gen_om(mod_path, weight_path, fmk: ModFMK, om_path, om_bin, om_params):
    cmd = [om_bin, "--model={}".format(mod_path),
           "--framework={}".format(fmk.value), "--output={}".format(om_path)]

    if fmk == ModFMK.CAFFE:
        cmd.append("--weight={}".format(weight_path))

    om_params = om_params if (isinstance(om_params, dict)) else {}
    for key, value in om_params.items():
        if key in ("model", "framework", "output", "weight"):
            mlog("om ignore {}:{} in json file".format(key, value),
                 logging.WARNING)
            continue
        cmd.append("--{}={}".format(key, value))

    mlog("start gen om: {}".format(cmd), level=logging.INFO)
    return subp.run(cmd)
