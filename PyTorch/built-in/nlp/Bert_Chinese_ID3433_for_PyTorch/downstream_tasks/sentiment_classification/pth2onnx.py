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


import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from bert4torch.snippets import get_pool_emb
from bert4torch.models import build_transformer_model, BaseModel


class Model(BaseModel):
    def __init__(self, config_path):
        super(Model, self).__init__()
        self.bert = build_transformer_model(config_path, checkpoint_path=None, with_pool=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], 2)

    def forward(self, token_ids, segment_ids):
        hidden_states, pooling = self.bert([token_ids, segment_ids])
        pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), 'cls')
        output = self.dropout(pooled_output)
        output = self.dense(output)
        return output
    
    def pad(self):
        # to pass pylint
        # add a pad function
        return self


def pth2onnx(args):
    # build model
    model = Model(args.config_path).to("cpu")
    model.load_weights(args.input_path, strict=False)

    # build data
    def build_input_data(shape):
        input_token = np.zeros(shape).astype('int64')
        input_segment = np.zeros(shape).astype('int64')
        return (torch.tensor(input_token, dtype=torch.long, device=device),
                 torch.tensor(input_segment, dtype=torch.long, device=device))

    dummy_input = build_input_data((args.batch_size, 256))
    input_names = ["token_ids", "segment_ids"]
    output_names = ["output"]
    torch.onnx.export(
        model,
        dummy_input,
        args.out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "token_ids": {0: "batch_size"},
            "segment_ids": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=11,
        verbose=True
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description='Sentiment Classification onnx export.')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='input path for pth model')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='save path for output onnx model')
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help='config path for export model')
    parser.add_argument('-s', '--seq_len', type=int, default=256,
                        help='max sequence length for output model')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='batch size of onnx model')
    args = parser.parse_args()
    args.out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(args.out_path))
    return args


if __name__ == '__main__':
    device = 'cpu'
    main_args = parse_arguments()
    pth2onnx(main_args)
