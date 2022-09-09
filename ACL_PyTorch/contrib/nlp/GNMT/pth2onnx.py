# Copyright 2022 Huawei Technologies Co., Ltd
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
import sys
import torch
import argparse
sys.path.append('./DeepLearningExamples/PyTorch/Translation/GNMT/')
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.models.gnmt import GNMT
import seq2seq.data.config as config

def run_pth2onnx(args):
    device = torch.device('cpu' if not args.use_cuda else 'cuda')
    checkpoint = torch.load(args.model, device)
    tokenizer = Tokenizer()
    tokenizer.set_state(checkpoint['tokenizer'])
    dtype = {
        'fp32': torch.FloatTensor,
        'fp16': torch.HalfTensor
    }

    model_config = checkpoint['model_config']
    model_config['batch_first'] = True
    model_config['vocab_size'] = tokenizer.vocab_size
    model_config['max_seq_len'] = args.max_seq_len
    model = GNMT(**model_config)
    model.load_state_dict(checkpoint['state_dict'])
    model.type(dtype[args.math])
    model = model.to(device)
    model.eval()
    
    input_encoder = torch.ones(1, args.max_seq_len, dtype=torch.int32, device=device)
    input_enc_len = torch.tensor([args.max_seq_len], dtype=torch.int32, device=device)

    bos = [[config.BOS]]
    input_decoder = torch.tensor(bos, dtype=torch.int32, device=device).view(-1, 1)

    input = (input_encoder, input_enc_len, input_decoder)

    torch.onnx.export(model, input, args.onnx_path,
                      input_names=['input_encoder', 'input_enc_len', 'input_decoder'],
                      output_names=['translation'],
                      export_params=True,
                      verbose=False,
                      opset_version=13)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./gnmt.pth')
    parser.add_argument('--onnx_dir', type=str, default='./')
    parser.add_argument('--math', default='fp32', choices=['fp16', 'fp32']) # float16 is supported only on gpu.
    parser.add_argument("--max_seq_len", type=int, default=30)
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()
    args.onnx_path = "{}/gnmt_msl{}.onnx".format(args.onnx_dir, args.max_seq_len)
    if not os.path.exists(args.onnx_dir):
        os.makedirs(args.onnx_dir)

    run_pth2onnx(args)


if __name__ == '__main__':
    main()
