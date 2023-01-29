# Copyright 2021 Huawei Technologies Co., Ltd
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


import transformers
import torch
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--pretrained_model', default='./model/pytorch_model.bin', type=str, required=False, help='模型起点路径')

    args = parser.parse_args()

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)

    model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model,config=model_config)
    model.eval()

    input_names = ["input_ids"]
    output_names = ["output"]
    input_ids = torch.ones(8, 512).long()
    dynamic_axes = {'input_ids': {0: '-1'},'output': {0: '-1'}}
    torch.onnx.export(model, 
                    input_ids,
                    'gpt2_dybs.onnx', 
                    input_names=input_names, 
                    dynamic_axes = dynamic_axes, 
                    output_names=output_names, 
                    opset_version=13)


if __name__ == '__main__':
    main()
