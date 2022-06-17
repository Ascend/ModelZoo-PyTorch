# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import re
import sys
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, default="./msame_bs1.txt")
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    if args.result_file.endswith('.json'):
        result_json = args.result_file
        with open(result_json, 'r') as f:
            content = f.read()
        tops = [i.get('value') for i in json.loads(content).get('value') if 'Top' in i.get('key')]
        print('om {} top1:{}'.format(result_json.split('_')[1].split('.')[0], tops[0]))
    elif args.result_file.endswith('.txt'):
        result_txt = args.result_file
        with open(result_txt, 'r') as f:
            content = f.read()
        txt_data_list = re.findall(r'Inference average time without first time:.*ms', content.replace('\n', ',') + ',')[-1]
        avg_time = txt_data_list.split(' ')[-2]
        fps = args.batch_size * 1000 / float(avg_time)
        print('310P bs{} fps:{:.3f}'.format(args.batch_size, fps))