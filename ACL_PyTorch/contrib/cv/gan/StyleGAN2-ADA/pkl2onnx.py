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

import pickle
import sys

sys.path.append('./model')
import torch
import argparse
import functools


def pkl2onnx(args):
    # Set up options
    bs = args.batch_size
    pkl_file = args.pkl_file
    onnx_file = args.onnx_file

    # Load pkl model
    with open(pkl_file, 'rb') as f:
        G = pickle.load(f)['G_ema']
    G.forward = functools.partial(G.forward, force_fp32=True)

    # Prepare input
    input_names = ["z"]
    output_names = ["image"]
    z = torch.randn([bs, G.z_dim])
    c = torch.empty([bs, 0], device='cpu')
    dummy_input = (z, c)

    # Prepare output name
    if onnx_file is None:
        onnx = "G_ema_onnx_bs{}.onnx".format(z.shape[0])
    else:
        onnx = onnx_file
    # export onnx
    torch.onnx.export(G, dummy_input, onnx,
                      input_names=input_names,
                      output_names=output_names,
                      do_constant_folding=False,
                      opset_version=11, 
                      verbose=True)


#
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file', type=str, default='./G_ema_bs8_8p_kimg1000.pkl')
    parser.add_argument('--onnx_file', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    pkl2onnx(args)
# ----------------------------------------------------------------------------
