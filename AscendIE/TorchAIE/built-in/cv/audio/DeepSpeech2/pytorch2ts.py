# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

import argparse

import torch

from deepspeech_pytorch.utils import load_model


parser = argparse.ArgumentParser(description='Deepspeech')
parser.add_argument('--ckpt_path', default='./an4_pretrained_v3.ckpt', type=str, help='infer out path')
parser.add_argument('--out_file', default='deepspeech_torchscript.pt', type=str, help='infer info path')
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cpu")
    model = load_model(device=device, model_path=args.ckpt_path)
    model.eval()
    model = model.to(device)
    print('Finished loading model!')

    dummy_input = torch.randn(1, 1, 161, 621).to(device)
    dummy_input2 = torch.tensor([621], dtype=torch.int32).to(device)
    output_file = args.out_file

    input_data = (dummy_input, dummy_input2)
    ts_model = torch.jit.trace(model, input_data)
    ts_model.save(output_file)
