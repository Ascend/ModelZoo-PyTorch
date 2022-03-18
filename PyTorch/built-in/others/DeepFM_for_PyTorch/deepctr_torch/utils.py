# -*- coding:utf-8 -*-

# Copyright 2020 Huawei Technologies Co., Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at# 
# 
#     http://www.apache.org/licenses/LICENSE-2.0# 
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from threading import Thread

import requests

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse


def check_version(version):
    """Return version of package on pypi.python.org using json."""

    def check(version):
        try:
            url_pattern = 'https://pypi.python.org/pypi/deepctr-torch/json'
            req = requests.get(url_pattern)
            latest_version = parse('0')
            version = parse(version)
            if req.status_code == requests.codes.ok:
                j = json.loads(req.text.encode('utf-8'))
                releases = j.get('releases', [])
                for release in releases:
                    ver = parse(release)
                    if ver.is_prerelease or  ver.is_postrelease:
                        continue
                    latest_version = max(latest_version, ver)
                if latest_version > version:
                    logging.warning(
                        '\nDeepCTR-PyTorch version {0} detected. Your version is {1}.\nUse `pip install -U deepctr-torch` to upgrade.Changelog: https://github.com/shenweichen/DeepCTR-Torch/releases/tag/v{0}'.format(
                            latest_version, version))
        except :
            print("Please check the latest version manually on https://pypi.org/project/deepctr-torch/#history")
            return

    Thread(target=check, args=(version,)).start()
