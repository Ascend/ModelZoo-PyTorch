# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


"""
Export onnx from ckpt
"""
import hydra
import os
import torch
from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.utils import load_model
import argparse

parser = argparse.ArgumentParser(description='Deepspeech')
parser.add_argument('--ckpt_path', default='./an4_pretrained_v3.ckpt', type=str, help='infer out path')
parser.add_argument('--out_file', default='deepspeech.onnx', type=str, help='infer info path')
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cpu")
    # device = torch.device("cuda" if EvalConfig.model.cuda else "cpu")
    model = load_model(device=device, model_path=args.ckpt_path)
    model.eval()
    model = model.to(device)
    print('Finished loading model!')
    # print(model)
    input_names = ["spect", "transcript"]
    output_names = ["out"]
    dynamic_axes = {'spect': {0: '-1'}}
    dummy_input = torch.randn(1, 1, 161, 621).to(device)
    dummy_input2 = torch.tensor([621], dtype=torch.int32).to(device)
    output_file = args.out_file
    torch.onnx.export(model, [dummy_input, dummy_input2], output_file, 
                      input_names=input_names, dynamic_axes=dynamic_axes, 
                      output_names=output_names, opset_version=11, verbose=True)
