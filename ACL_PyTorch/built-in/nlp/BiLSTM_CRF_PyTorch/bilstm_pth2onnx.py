# Copyright 2023 Huawei Technologies Co., Ltd
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


import sys
import argparse
from pathlib import Path

import torch
import numpy as np

import config
from model import NERModel
from preprocess import Preprocessor
from common import load_model


def pth2onnx(checkpoint, model_cfg, dummy_input, 
             output_path='out.onnx'):
    
    # build pytorch model
    model = NERModel(**model_cfg)
    model = load_model(model, model_path=checkpoint)
    model.eval()

    dynamic_axes = {'ids': {0: 'bs'}, 
                    'mask':{0: 'bs'}, 
                    'features': {0: 'bs'}}

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['ids', 'mask'],
        output_names=['features'],
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=12,
        verbose=False,
        dynamic_axes=dynamic_axes
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,   
        help="path to checkpoint.")
    parser.add_argument("--vocab", type=str, 
        default='CLUENER2020/bilstm_crf_pytorch/dataset/cluener/vocab.pkl', 
        help='path to om vocab file (.pkl)')
    parser.add_argument("--max_seq_len", type=int, default=50,  
        help="path to checkpoint.")
    parser.add_argument("--output", type=str, default='./out.onnx',  
        help="path to save onnx file.")
    args = parser.parse_args()

    # dummy input
    preprocessor = Preprocessor(max_seq_len=50, vocab_path=args.vocab)
    text = "工信部提醒岁末当心刷卡类短信诈骗"
    input_ids, input_mask, _ = preprocessor.process_text(text)

    model_cfg = dict(
        vocab_size=len(preprocessor.vocab), 
        embedding_size=128,
        hidden_size=384,
        device=torch.device("cpu"),
        label2id=config.label2id
    )

    # export onnx file
    pth2onnx(args.checkpoint, model_cfg, 
             (input_ids, input_mask),
             output_path=args.output)
    print(f'Successfully exported ONNX model: {args.output}')

    # export transitions of CRF
    state_dict = torch.load(args.checkpoint)['state_dict']
    transitions = state_dict['crf.transitions'].numpy()
    save_path = str(Path(args.output).parent / 'transitions.npy')
    np.save(save_path, transitions)
    print(f'Successfully exported crf.transitions: {save_path}')


if __name__ == "__main__":
    main()
