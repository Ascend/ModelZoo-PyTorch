# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
import torch_aie
from torch_aie import _enums
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='./config.json', type=str, required=False,
                        help='model config json path')
    parser.add_argument('--pretrained_model', default='./model/pytorch_model.bin',
                        type=str, required=False, help='model checkpoint path')
    parser.add_argument('--batch_size', default=1, type=int,
                        required=False, help='batch size')
    parser.add_argument('--device', default=0, type=int,
                        required=False, help='npu device')
    parser.add_argument('--optimization_level', default=0, type=int,
                        required=False, help='optimization_level')

    args = parser.parse_args()
    device = args.device
    batch_size = args.batch_size
    optimization_level = args.optimization_level
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(
        args.model_config)
    model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(
        args.pretrained_model, config=model_config)
    model.eval()
    aie_model_path = "gpt2_bs" + str(batch_size) + ".pth"

    torch_aie.set_device(device)

    accept_size = [batch_size, 512]
    dummy_input = torch.ones(accept_size).long()
    with torch.inference_mode():
        jit_model = torch.jit.trace(model, dummy_input)
        aie_input_spec = [torch_aie.Input(
            accept_size, dtype=torch_aie.dtype.INT64),]
        aie_model = torch_aie.compile(
            jit_model,
            inputs=aie_input_spec,
            precision_policy=_enums.PrecisionPolicy.FP16,
            truncate_long_and_double=True,
            require_full_compilation=False,
            allow_tensor_replace_int=False,
            min_block_size=3,
            torch_executed_ops=[],
            soc_version="Ascend310P3",
            optimization_level=optimization_level)
        aie_model.save(aie_model_path)

if __name__ == '__main__':
    main()
