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
import numpy as np
import argparse
import torch
import random
import os
import modeling

def make_train_dummy_input():
    org_input_ids = torch.ones(8, 512).long()
    org_token_type_ids = torch.ones(8, 512).long()
    org_input_mask = torch.ones(8, 512).long()
    return (org_input_ids, org_token_type_ids, org_input_mask)

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")   
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="use mixed-precision")
    args = parser.parse_args()
    MODEL_ONNX_PATH = "./bert_base_batch_8.onnx"
    OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
    config = modeling.BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = modeling.BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu')["model"])
    model.to("cpu")
    if args.fp16:
        model.half()
    
    model.eval()
    org_dummy_input = make_train_dummy_input()
    output = torch.onnx.export(model,
                               org_dummy_input,
                               MODEL_ONNX_PATH,
                               verbose=True,
                               operator_export_type=OPERATOR_EXPORT_TYPE,
                               input_names=['input_ids', 'token_type_ids', 'attention_mask'],
                               output_names=['output'],
                               dynamic_axes={'input_ids': [0], 'token_type_ids': [0], 'attention_mask': [0]},
                               opset_version=11
                               )
    print("Export of torch_model.onnx complete!")