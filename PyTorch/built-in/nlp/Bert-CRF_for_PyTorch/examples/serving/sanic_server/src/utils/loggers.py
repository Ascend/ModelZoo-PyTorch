# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================

# -*- coding: utf-8 -*-

import logging
import logging.config
from typing import Optional, Text
from src.utils.configs import Configuration

TRACE_LOG = "tracelogger"
ERROR_LOG = "errorlogger"
OUT_LOG = "outlogger"
logger = logging.getLogger(__name__)
LOG_PATH_KEY = "log_path"


def configure_file_logging(config_path: Optional[Text]):
    if config_path is None:
        return

    dict = Configuration.read_config_file(config_path + "/logger.yml")
    import os
    if LOG_PATH_KEY in dict:
        log_path = dict[LOG_PATH_KEY]
        if not os.path.exists(log_path):
            os.makedirs(log_path)  # 创建路径
        dict.pop(LOG_PATH_KEY)
    # logging.config.dictConfig(codecs.open(config_path + "\logger.yml", 'r', 'utf-8').read())
    logging.config.dictConfig(dict)


def get_trace_log():
    return logging.getLogger(TRACE_LOG)


def get_error_log():
    return logging.getLogger(ERROR_LOG)


def get_out_log():
    return logging.getLogger(OUT_LOG)


if __name__ == '__main__':
    pass

