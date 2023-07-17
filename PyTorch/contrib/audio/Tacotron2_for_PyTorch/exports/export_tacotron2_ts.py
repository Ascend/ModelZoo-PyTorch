# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
from inference import load_and_setup_model


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--tacotron2', type=str, required=True,
                        help='full path to the Tacotron2 model checkpoint file')

    parser.add_argument('-o', '--output', type=str, default="trtis_repo/tacotron/1/model.pt",
                        help='filename for the Tacotron 2 TorchScript model')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with mixed precision')

    return parser


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args = parser.parse_args()

    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     amp_run=args.fp16, cpu_run=False,
                                     forward_is_infer=True)

    jitted_tacotron2 = torch.jit.script(tacotron2)

    torch.jit.save(jitted_tacotron2, args.output)


if __name__ == '__main__':
    main()
