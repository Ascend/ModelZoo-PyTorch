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

from sanic import response
from src.utils.trace_log import TraceLog
from src.utils import loggers
import traceback
import uuid
import time


async def setup_model(app, loop):
    loggers.get_out_log().info("----------setup model-------------")
    global model
    model = app.modelSortLightGBM
    model.load_model()

    loggers.get_out_log().info("----------done setup model-------------")


async def health_check(request):
    return response.json({"status": "ok"})


async def process_rec_info(request):
    tracelog = TraceLog()
    try:
        data = request.json
        rid = data.get("requestid", uuid.uuid4().hex)
        input_sent = data.get("input")

        # todo: log params
        tracelog.apiVersion(1)
        tracelog.requestId(rid)
        tracelog.inputSent(input_sent)
        tracelog.start_log()

        # 模型推理
        all_start = time.time()
        cost_detail = {}
        finalresult = await model.process(input_sent)
        cost_detail['all_process'] = (time.time() - all_start) * 1000
        tracelog.costDetail(cost_detail)

        tracelog.modelResults(finalresult)
        tracelog.modelResultsLen(len(finalresult))

        ret = {
            "code": 0,
            "requestid": rid,
            "errmsg": "",
            "total": len(finalresult),
            "recResults": finalresult
            }
    except Exception as e:
        loggers.get_error_log().error("error occur in recommand infos {}".format(traceback.format_exc()))
        t = "{}".format(e)
        ret = {
            "code": -1,
            "requestid": rid,
            "errmsg": f"{t}",
            "total": 0,
            "recResults": [{}]
        }
        tracelog.exception(t)

    tracelog.responseEntity(ret)
    tracelog.end_log()
    return response.json(ret)
	
