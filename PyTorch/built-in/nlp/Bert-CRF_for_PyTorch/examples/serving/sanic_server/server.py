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

from sanic import Sanic
from typing import Optional, Text
import src.config.constants as constants
import src.utils.loggers as loggers
import json


def create_app(confs: Optional[Text] = None):

    from src.utils.configs import Configuration
    Configuration.configurations = Configuration.read_config_file(confs + '/configurations.yml')
    loggers.get_out_log().info("configurations: {}.".format(json.dumps(Configuration.configurations)))

    from src.utils.loggers import configure_file_logging
    configure_file_logging(confs)

    app = Sanic(__name__)
    register_view(app)

    return app


def register_view(app):
    from src.view.view import setup_model, health_check, process_rec_info
    from src.model.model import BertModel

    app.modelSortLightGBM = BertModel()

    app.register_listener(setup_model, "before_server_start")
    # app.add_task() # 一些后台任务

    app.add_route(handler=health_check, uri="/", methods={"GET"})
    # get请求展示报错情况，日志如何记录。 post请求展示正常情况。
    app.add_route(handler=process_rec_info, uri="/recommendinfo", methods={"POST"})


def start_server(confs: Optional[Text] = None, port: int = constants.DEFAULT_SERVER_PORT):
    server = create_app(confs)
    protocol = "http"
    loggers.get_out_log().info(
        "Starting server on "
        "{}".format(constants.DEFAULT_SERVER_FORMAT.format(protocol, port))
    )
    server.run(host='0.0.0.0', port=port, debug=False, workers=1)


if __name__ == "__main__":
    start_server(confs='E:/Github/bert4torch/examples/serving/sanic_server/conf')
