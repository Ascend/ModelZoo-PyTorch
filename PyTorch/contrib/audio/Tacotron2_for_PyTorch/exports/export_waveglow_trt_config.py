# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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


import os
import argparse


def parse_args(parser):
    """
        Parse commandline arguments.
    """
    parser.add_argument("--trtis_model_name",
                        type=str,
                        default='waveglow',
                        help="exports to appropriate directory for TRTIS")
    parser.add_argument("--trtis_model_version",
                        type=int,
                        default=1,
                        help="exports to appropriate directory for TRTIS")
    parser.add_argument('--fp16', action='store_true',
                        help='inference with mixed precision')
    return parser


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch WaveGlow TRTIS config exporter')
    parser = parse_args(parser)
    args = parser.parse_args()

    # prepare repository
    model_folder = os.path.join('./trtis_repo', args.trtis_model_name)
    version_folder = os.path.join(model_folder, str(args.trtis_model_version))
    if not os.path.exists(version_folder):
        os.makedirs(version_folder)

    # build the config for TRTIS
    config_filename = os.path.join(model_folder, "config.pbtxt")
    config_template = r"""
name: "{model_name}"
platform: "tensorrt_plan"
default_model_filename: "waveglow_fp16.engine"

max_batch_size: 1

input {{
  name: "mel"
  data_type: {fp_type}
  dims: [80, -1, 1]
}}
input {{
  name: "z"
  data_type: {fp_type}
  dims: [8, -1, 1]
}}
output {{
  name: "audio"
  data_type: {fp_type}
  dims: [-1]
}}
"""

    config_values = {
        "model_name": args.trtis_model_name,
        "fp_type": "TYPE_FP16" if args.fp16 else "TYPE_FP32"
    }

    with open(model_folder + "/config.pbtxt", "w") as file:
        final_config_str = config_template.format_map(config_values)
        file.write(final_config_str)


if __name__ == '__main__':
    main()
