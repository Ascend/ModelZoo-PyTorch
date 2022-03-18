# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
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