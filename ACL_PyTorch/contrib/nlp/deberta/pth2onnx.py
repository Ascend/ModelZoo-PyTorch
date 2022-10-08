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
# ============================================================================

import numpy as np
import os
import sys
import torch
import argparse
sys.path.append('./DeBERTa/')
from DeBERTa.apps.models import SequenceClassificationModel

def create_model(args, num_labels, model_class_fn):
    # Prepare model
    init_model = args.init_model
    model_config = args.config
    model = model_class_fn(init_model, model_config, num_labels=num_labels, \
        drop_out=0.1, pre_trained = None)
    return model

def get_labels():
    """See base class."""
    return ["contradiction", "neutral", "entailment"]

def run_pth2onnx(args):
    label_list = get_labels()
    device = torch.device('cpu')
    model = create_model(args,len(label_list), SequenceClassificationModel.load_model)
    model = model.to(device)
    model.eval()

    input_ids = torch.zeros([1, 256], dtype=torch.int32).to(device)
    input_mask = torch.zeros([1, 256], dtype=torch.int32).to(device)

    out = model(input_ids, input_mask)
    dynamic_axes={'input_ids': {0: '-1'}, 'input_mask': {0: '-1'}, 'logits':  {0: '-1'}}

    torch.onnx.export(model, (input_ids, input_mask), args.onnx_path,
                     input_names=["input_ids", "input_mask"],
                     output_names=["logits"],
                     opset_version=12,
                     dynamic_axes = dynamic_axes)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_model', type=str, default='./pytorch.model-018407.bin')
    parser.add_argument('--config', type=str, default='./model_config.json')
    parser.add_argument('--onnx_path', type=str, default='./dynamic.onnx')
    args = parser.parse_args()

    run_pth2onnx(args)