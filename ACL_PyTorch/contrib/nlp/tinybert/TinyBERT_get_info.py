# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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

"""Run TinyBERT on SST-2."""
import argparse

def main():
    """output：info file"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        default=1,
                        type=int,
                        required=True)
    parser.add_argument("--output_path",
                        default='./bert_bin/',
                        type=str,
                        required=True,
                        help='The output dir of info file.')
    args = parser.parse_args()
    test_num = 872
    base_path = args.output_path
    with open('./TinyBERT.info', 'w') as f:
        for i in range(test_num):
            ids_name = base_path + 'input_ids_{}.bin'.format(i)
            segment_name = base_path + 'segment_ids_{}.bin'.format(i)
            mask_name = base_path + 'input_mask_{}.bin'.format(i)
            f.write(str(i) + ' ' + ids_name)
            f.write('\n')
            f.write(str(i) + ' ' + segment_name)
            f.write('\n')
            f.write(str(i) + ' ' + mask_name)
            f.write('\n')

if __name__ == "__main__":
    main()