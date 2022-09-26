# Copyright 2021 Huawei Technologies Co., Ltd
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

import os.path as osp

from mmcv.utils import build_from_cfg

from mmdet.datasets.builder import PIPELINES


def test_default_format_bundle():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../data'),
        img_info=dict(filename='color.jpg'))
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)
    bundle = dict(type='DefaultFormatBundle')
    bundle = build_from_cfg(bundle, PIPELINES)
    results = load(results)
    assert 'pad_shape' not in results
    assert 'scale_factor' not in results
    assert 'img_norm_cfg' not in results
    results = bundle(results)
    assert 'pad_shape' in results
    assert 'scale_factor' in results
    assert 'img_norm_cfg' in results
