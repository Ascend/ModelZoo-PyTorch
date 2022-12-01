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

import os
import sys
import numpy as np

from magiconnx import OnnxGraph, INITIALIZER
from magiconnx.optimize.optimizer_manager import OptimizerManager


if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    onnx_graph = OnnxGraph(input_path)
    
    # fix for precision overflow
    muls = onnx_graph.get_nodes('Mul')
    for mul in muls:
        if len(onnx_graph.get_next_nodes(mul.name)) == 24:
            mul_node = mul
    for input_name in mul_node.inputs:
        try:
            node = onnx_graph[input_name]
            if node.op_type == INITIALIZER:
                node.value = np.array(-10000).astype('float32')
        except KeyError:
            pass

    # bert big kernel
    optimize_manager_bert = OptimizerManager(onnx_graph, optimizers=["BertBigKernelOptimizer"])
    optimize_manager_bert.apply()
    onnx_graph.save(save_path)

