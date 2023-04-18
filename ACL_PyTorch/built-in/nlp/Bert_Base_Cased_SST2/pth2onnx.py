# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification


def build_model(model_path):
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=2,
        finetining_task='sst2',
        cache_dir=None,
        use_auth_token=None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        cache_dir=None,
        use_auth_token=None,
        ignore_mismatched_sizes=False,
    )
    model.eval()
    return model


def export_onnx(model_dir, save_path):
    # build model
    model = build_model(model_dir)

    # build data
    input_data = (
        torch.randint(0, 10000, (1, 128)),
        torch.randint(0, 2, (1, 128))
    )
    input_names = ["input_ids", "attention_mask"]
    output_names = ["label"]
    dynamic_axes = {
        "input_ids": {0: "-1", 1: "-1"},
        "attention_mask": {0: "-1", 1: "-1"},
        "label": {0: "-1"}
    }

    # export onnx model
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.onnx.export(
        model,
        input_data,
        save_path,
        dynamic_axes=dynamic_axes,
        opset_version=11,
        input_names=input_names,
        output_names=output_names
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True,
                        help='path of the folder of model pth and config')
    parser.add_argument('--save_path', required=True,
                        help='path of the onnx model')
    args = parser.parse_args()
    export_onnx(args.model_dir, args.save_path)
