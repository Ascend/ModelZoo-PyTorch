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

import torch
import argparse
import os

from tacotron2_common.utils import ParseFromConfigFile
from inference import load_and_setup_model


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--waveglow', type=str, required=True,
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory for the exported WaveGlow ONNX model')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with AMP')
    parser.add_argument('-s', '--sigma-infer', default=0.6, type=float)

    parser.add_argument('--config-file', action=ParseFromConfigFile,
                         type=str, help='Path to configuration file')

    return parser


def export_onnx(parser, args):

    waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    fp16_run=args.fp16, cpu_run=False,
                                    forward_is_infer=False)

    # 80 mel channels, 620 mel spectrograms ~ 7 seconds of speech
    mel = torch.randn(1, 80, 620).cuda()
    stride = 256 # value from waveglow upsample
    n_group = 8
    z_size2 = (mel.size(2)*stride)//n_group
    z = torch.randn(1, n_group, z_size2).cuda()

    if args.fp16:
        mel = mel.half()
        z = z.half()
    with torch.no_grad():
        # run inference to force calculation of inverses
        waveglow.infer(mel, sigma=args.sigma_infer)

        # export to ONNX
        if args.fp16:
            waveglow = waveglow.half()

        waveglow.forward = waveglow.infer_onnx

        opset_version = 12

        output_path = os.path.join(args.output, "waveglow.onnx")
        torch.onnx.export(waveglow, (mel, z), output_path,
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=["mel", "z"],
                          output_names=["audio"],
                          dynamic_axes={"mel":   {0: "batch_size", 2: "mel_seq"},
                                        "z":     {0: "batch_size", 2: "z_seq"},
                                        "audio": {0: "batch_size", 1: "audio_seq"}})


def main():

    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    export_onnx(parser, args)


if __name__ == '__main__':
    main()
