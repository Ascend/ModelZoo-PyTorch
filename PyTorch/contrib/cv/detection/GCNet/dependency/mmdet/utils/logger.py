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

import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger
