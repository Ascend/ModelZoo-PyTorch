# Copyright 2020 Huawei Technologies Co., Ltd
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

# 3d_nested_unet_preprocess.py
import sys
import os
import time
import pdb
import argparse
from nnunet.inference import infer_path


def main():
    # pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp1', '--file_path1', help='INFERENCE_INPUT_FOLDER', required=True, default='/home/hyp/environment/input/')
    parser.add_argument('-fp2', '--file_path2', help='INFERENCE_OUTPUT_FOLDER', required=True, default='/home/hyp/environment/output/')
    parser.add_argument('-fp3', '--file_path3', help='INFERENCE_SHAPE_PATH', required=True, default='/home/hyp/environment/')
    args = parser.parse_args()
    python_file = infer_path.__file__
    fp1 = args.file_path1
    fp2 = args.file_path2
    fp3 = args.file_path3
    lines = []
    print('尝试读取：', python_file)
    file = open(python_file, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    print('尝试修改路径')
    with open(python_file, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.startswith('INFERENCE_INPUT_FOLDER'):
                line = 'INFERENCE_INPUT_FOLDER = ' + '\'' + str(fp1) + '\'' + '\n'
            if line.startswith('INFERENCE_OUTPUT_FOLDER'):
                line = 'INFERENCE_OUTPUT_FOLDER = ' + '\'' + str(fp2) + '\'' + '\n'
            if line.startswith('INFERENCE_SHAPE_PATH'):
                line = 'INFERENCE_SHAPE_PATH = ' + '\'' + str(fp3) + '\'' + '\n'
            f.write(line)
        print('正在修改：', python_file)
        print('INFERENCE_INPUT_FOLDER =', fp1)
        print('INFERENCE_OUTPUT_FOLDER=', fp2)
        print('INFERENCE_SHAPE_PATH   =', fp3)
    f.close()
    print('修改完成')


if __name__ == "__main__":
    main()
    print('main end')

