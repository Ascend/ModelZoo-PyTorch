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

import torch
import torch_aie

# After "python3 setup.py install --user", you should find this .so file under generated "build" directory
torch.ops.load_library(
    "./build/lib.linux-x86_64-3.9/custom_converter.cpython-39-x86_64-linux-gnu.so"
)


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        x = torch.cumsum(x, 0)
        return x

model = MyModel().eval()
inputs = torch.randint(1, 10, (5, 5)).type(torch.float)
tracedModel = torch.jit.trace(model, inputs)

compile_input = [torch_aie.Input((5, 5))]

torch_aie.set_device(0)

compiled_module = torch_aie.compile(
    tracedModel,
    inputs=compile_input,
    precision_policy=torch_aie.PrecisionPolicy.FP16,
    truncate_long_and_double=True,
    require_full_compilation=False,
    allow_tensor_replace_int=False,
    min_block_size=1,
    torch_executed_ops=[],
    soc_version="Ascend310P3",
    optimization_level=0
)

inputs = torch.randint(1, 10, (5, 5)).type(torch.float)
results_aie = compiled_module.forward(inputs)
results_torch = tracedModel.forward(inputs)
print("input:\n", inputs)
print("aie output:\n", results_aie.to("cpu"))
print("torch output:\n", results_torch)