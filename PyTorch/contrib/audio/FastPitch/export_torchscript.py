# Copyright 2020 Huawei Technologies Co., Ltd
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

import argparse
import torch
from inference import load_and_setup_model


def parse_args(parser):
    parser.add_argument('--generator-name', type=str, required=True,
                        choices=('Tacotron2', 'FastPitch'), help='model name')
    parser.add_argument('--generator-checkpoint', type=str, required=True,
                        help='full path to the generator checkpoint file')
    parser.add_argument('-o', '--output', type=str, default="trtis_repo/tacotron/1/model.pt",
                        help='filename for the Tacotron 2 TorchScript model')
    parser.add_argument('--amp', action='store_true',
                        help='inference with AMP')
    return parser


def main():
    parser = argparse.ArgumentParser(description='Export models to TorchScript')
    parser = parse_args(parser)
    args = parser.parse_args()

    model = load_and_setup_model(
        args.generator_name, parser, args.generator_checkpoint,
        args.amp, device='cpu', forward_is_infer=True, polyak=False,
        jitable=True)
    
    torch.jit.save(torch.jit.script(model), args.output)
    

if __name__ == '__main__':
    main()

    
