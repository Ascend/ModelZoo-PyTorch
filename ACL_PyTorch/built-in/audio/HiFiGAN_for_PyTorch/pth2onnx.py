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
import json
from env import AttrDict
import torch

from models import Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--checkpoint_file', type=str, default='generator_v1')
    parser.add_argument('--config_file', type=str, default='config_v1.json')
    args = parser.parse_args()

    # load config
    with open(args.config_file) as f:
        data = f.read()
    json_config = json.loads(data)

    # load model
    generator = Generator(AttrDict(json_config))
    state_dict_g = torch.load(args.checkpoint_file, map_location='cpu')
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    # export onnx
    print("Starting export hifigan ……")
    dummy_input = torch.randn(1, 80, 1024)
    torch.onnx.export(generator, dummy_input, os.path.join(args.output_dir, args.checkpoint_file + '.onnx'),
                      opset_version=11, do_constant_folding=True,
                      input_names=["mel_spec"],
                      output_names=["wavs"],
                      dynamic_axes={"mel_spec": {0: "batch_size", 2: "mel_len"}})
    print("export hifigan success")
