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
import torch
import argparse
from transformers import Wav2Vec2ForCTC

if __name__ == '__main__':
    '''
    Using Example:

    python export_onnx.py --model_save_dir=./models
    '''

    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_dir', required=True)
    opt = parser.parse_args()
    model_save_dir = opt.model_save_dir

    # 加载模型
    print('[INFO] Start loading model.')
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    print('[INFO] Model load successfully.')

    # 创建目录
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # 导出onnx
    print('[INFO] Export to onnx ...')
    model.eval()
    input = torch.ones(1, 100000, dtype=torch.float)
    torch.onnx.export(model,
                      input,
                      os.path.join(model_save_dir, 'wav2vec2-base-960h.onnx'),
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={
                          'input': {
                              0: 'batch_size',
                              1: 'data_len'
                          },
                          'output': {
                              0: 'batch_size',
                              1: 'data_len'
                          }
                      },
                      opset_version=11,
                      enable_onnx_checker=True)
    print('[INFO] Done!')
