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

import sys
import os
import argparse
import re

def parse(args):
    """[Acquisition and processing of performance parameters]

    Args:
        args ([argparse]): [input parameters]
    """
    if args.file_type == 'txt':
        if isinstance(args.file_name, list):
            txt_list = args.file_name
            bin_list = args.bin_path
        else:
            txt_list = [args.file_name]
            bin_list = [args.bin_path]
        fps_all = 0
        bin_len_all = 0 # all bin numbers in all bin path
        for k in range(len(txt_list)):
            result_txt = txt_list[k]
            bin_len = len(os.listdir(bin_list[k])) # bin numbers in one bin path
            bin_len_all = bin_len_all + bin_len
            with open(result_txt, 'r') as f:
                content = f.read()
            txt_data_list = [i.strip() for i in re.findall(r':(.*?),', content.replace('\n', ',') + ',')]
            fps = float(txt_data_list[7].replace('samples/s', '')) * 4
            # print(fps, bin_len)
            fps_all = fps_all + fps * bin_len
        print("====310 performance data====")
        print('310 bs{} fps:{}'.format(result_txt.split('_')[3], fps_all / bin_len_all))
    elif args.file_type == 'json':
        print(args.file_name)
        print(args.hhh)
        result_json = args.file_name[0]
        with open(result_json, 'r') as f:
            content = f.read()
        print(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse')
    parser.add_argument('--file_type', default='txt', type=str, help='file_type')
    parser.add_argument('--file_name', nargs='+', type=str, help='file_name')
    parser.add_argument('--bin_path', nargs='+', type=str, help='bin file path inferred by benchmark')
    args = parser.parse_args()
    parse(args)
