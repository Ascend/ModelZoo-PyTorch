# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import sys
from pathlib import Path

import pytest

import mmcv


def test_is_filepath():
    assert mmcv.is_filepath(__file__)
    assert mmcv.is_filepath('abc')
    assert mmcv.is_filepath(Path('/etc'))
    assert not mmcv.is_filepath(0)


def test_fopen():
    assert hasattr(mmcv.fopen(__file__), 'read')
    assert hasattr(mmcv.fopen(Path(__file__)), 'read')


def test_check_file_exist():
    mmcv.check_file_exist(__file__)
    if sys.version_info > (3, 3):
        with pytest.raises(FileNotFoundError):  # noqa
            mmcv.check_file_exist('no_such_file.txt')
    else:
        with pytest.raises(IOError):
            mmcv.check_file_exist('no_such_file.txt')


def test_scandir():
    folder = osp.join(osp.dirname(__file__), 'data/for_scan')
    filenames = ['a.bin', '1.txt', '2.txt', '1.json', '2.json']

    assert set(mmcv.scandir(folder)) == set(filenames)
    assert set(mmcv.scandir(folder, '.txt')) == set(
        [filename for filename in filenames if filename.endswith('.txt')])
    assert set(mmcv.scandir(folder, ('.json', '.txt'))) == set([
        filename for filename in filenames
        if filename.endswith(('.txt', '.json'))
    ])
    assert set(mmcv.scandir(folder, '.png')) == set()
    with pytest.raises(TypeError):
        mmcv.scandir(folder, 111)
