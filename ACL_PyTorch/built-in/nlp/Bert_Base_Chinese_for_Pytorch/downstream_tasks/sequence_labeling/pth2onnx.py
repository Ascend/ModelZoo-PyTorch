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
import argparse
import numpy as np
import torch
import torch.nn as nn
from bert4torch.layers import CRF
from bert4torch.models import build_transformer_model, BaseModel


class Model(BaseModel):
    def __init__(self, config_path):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=None, segment_vocab_size=0)
        # embedding_dims:768, len_categories: 7
        self.fc = nn.Linear(768, 7)  # 包含首尾
        self.crf = CRF(7)

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        emission_score = self.fc(sequence_output)  # [btz, seq_len, tag_size]
        attention_mask = token_ids.gt(0).long()
        return emission_score, attention_mask


def build_model(config_path, checkpoint_path):
    model = Model(config_path).to("cpu")
    model.load_weights(checkpoint_path, strict=False)
    return model
 
 
def pth2onnx(args):
    # build model
    model = build_model(args.config_path, args.input_path)

    # build data
    dummy_input = torch.randint(1, 1024, (1, 256))
    input_names = ["token_ids"]
    output_names = ["emission_score", "attention_mask"]
    torch.onnx.export(
        model,
        dummy_input,
        args.out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "token_ids": {0: "batch_size"},
            "emission_score": {0: "batch_size"},
            "attention_mask": {0: "batch_size"}
        },
        opset_version=11,
        verbose=True
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description='Bert_Base_Chinese onnx export.')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='input path for pth model')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='save path for output onnx model')
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help='config path for export model')
    parser.add_argument('-s', '--seq_len', type=int, default=256,
                        help='max sequence length for output model')
    arguments = parser.parse_args()
    arguments.out_path = os.path.abspath(arguments.out_path)
    os.makedirs(os.path.dirname(arguments.out_path), exist_ok=True)
    return arguments


if __name__ == '__main__':
    main_args = parse_arguments()
    pth2onnx(main_args)
