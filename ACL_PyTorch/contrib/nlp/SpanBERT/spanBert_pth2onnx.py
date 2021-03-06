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
import sys
sys.path.append(r"./SpanBERT/code")
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert import modeling
from collections import OrderedDict
import numpy as np
import argparse
import torch


def make_train_dummy_input():
    org_input_ids = torch.ones(1, 512).long()
    org_token_type_ids = torch.ones(1, 512).long()
    org_input_mask = torch.ones(1, 512).long()

    return (org_input_ids, org_token_type_ids, org_input_mask)


def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument("--checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining") 
    parser.add_argument("--onnx_path",
                        default='./spanBert_dynamicbs.onnx',
                        type=str)
    parser.add_argument('--fp16',
                        action='store_true',
                        default=False,
                        help="use mixed-precision")
    args = parser.parse_args()
    MODEL_ONNX_PATH = args.onnx_path
    OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
    config = modeling.BertConfig.from_json_file(args.config_file)
    model = modeling.BertForQuestionAnswering(config) 
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint = proc_nodes_module(checkpoint)
    model.load_state_dict(checkpoint)
    model.to("cpu")
    if args.fp16:
        model.half()
    dynamic_axes = {'input_ids': {0: '-1'}, 'token_type_ids': {0: '-1'},'attention_mask': {0: '-1'},'output': {0: '-1'}} 
    model.eval()
    org_dummy_input = make_train_dummy_input()
    output = torch.onnx.export(model,
                               org_dummy_input,
                               MODEL_ONNX_PATH,
                               verbose=True,
                               operator_export_type=OPERATOR_EXPORT_TYPE,
                               input_names=['input_ids', 'token_type_ids', 'attention_mask'],
                               output_names=['output'],
                               opset_version=11,
                               dynamic_axes = dynamic_axes
                               )
    print("Export of spanBert_dynamicbs.onnx complete!")
