# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
import onnx
from parse import parse_args


def main():
    paser = parse_args(True, True)
    opt = paser.initialize()
    # Mode attr of Pad only supports constant, current is reflect ."
    model = onnx.load(opt.onnx_path + opt.model_ga_onnx_name)
    max_idx = len(model.graph.node)
    for i in range(max_idx):
        for k in range(len(model.graph.node[i].attribute)):

            if (model.graph.node[i].attribute[k].name == 'mode'):
                model.graph.node[i].attribute[k].s = b'constant'
                print(model.graph.node[i].attribute[k].s)

    onnx.checker.check_model(model)
    onnx.save(model, opt.onnx_path + opt.model_gb_onnx_name)

    model = onnx.load(opt.onnx_path + opt.model_gb_onnx_name)
    max_idx = len(model.graph.node)
    for i in range(max_idx):
        # if(model.graph.node[i].attribute[0].name=='Pad'):
        for k in range(len(model.graph.node[i].attribute)):

            if (model.graph.node[i].attribute[k].name == 'mode'):
                model.graph.node[i].attribute[k].s = b'constant'
                print(model.graph.node[i].attribute[k].s)
    onnx.checker.check_model(model)
    onnx.save(model, opt.onnx_path + opt.model_ga_onnx_name)


if __name__ == '__main__':
    main()
