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
import os
import os.path as osp
import tempfile

import mmcv.utils.logging

from mmcls.utils import get_root_logger, load_json_log


def test_get_root_logger():
    # Reset the initialized log
    mmcv.utils.logging.logger_initialized = {}
    with tempfile.TemporaryDirectory() as tmpdirname:
        log_path = osp.join(tmpdirname, 'test.log')

        logger = get_root_logger(log_file=log_path)
        message1 = 'adhsuadghj'
        logger.info(message1)

        logger2 = get_root_logger()
        message2 = 'm,tkrgmkr'
        logger2.info(message2)

        with open(log_path, 'r') as f:
            lines = f.readlines()
            assert message1 in lines[0]
            assert message2 in lines[1]

        assert logger is logger2

        handlers = list(logger.handlers)
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        os.remove(log_path)


def test_load_json_log():
    log_path = 'tests/data/test.logjson'
    log_dict = load_json_log(log_path)

    # test log_dict
    assert set(log_dict.keys()) == set([1, 2, 3])

    # test epoch dict in log_dict
    assert set(log_dict[1].keys()) == set(
        ['iter', 'lr', 'memory', 'data_time', 'time', 'mode'])
    assert isinstance(log_dict[1]['lr'], list)
    assert len(log_dict[1]['iter']) == 4
    assert len(log_dict[1]['lr']) == 4
    assert len(log_dict[2]['iter']) == 3
    assert len(log_dict[2]['lr']) == 3
    assert log_dict[3]['iter'] == [10, 20]
    assert log_dict[3]['lr'] == [0.33305, 0.34759]
