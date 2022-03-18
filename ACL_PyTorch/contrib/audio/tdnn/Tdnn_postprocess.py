# -*- coding: utf-8 -*-
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

import os
import re
import argparse
import numpy as np
label = {0:'3526', 1:'7312', 2:'1088', 3:'32', 4:'460', 5:'7859', 6:'118', 7:'6848', 8:'8629', 9:'163', 10:'2416', 11:'3947', 12:'332', 13:'19', 14:'6272', 15:'7367', 16:'1898', 17:'3664', 18:'2136', 19:'4640', 20:'1867', 21:'1970', 22:'4680', 23:'226', 24:'5789', 25:'3242', 26:'667', 27:'1737'}

if __name__ == '__main__':
    '''
    参数说明：
        --data_info: 数据集信息
        --result_dir: 二进制推理结果目录
    '''

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_info', default='mini_librispeech_test.info')
    parser.add_argument('--result_dir', default='result')

    opt = parser.parse_args()
    error = 0
    total = 0

    with open('mini_librispeech_test.info', 'r') as f:
        for line in f.readlines():
            # line format example
            # 0 mini_librispeech_test_bin/4680-16042-0024.bin (1,1600,23)
            split = line.split(' ')
            index = split[0]
            input_file = split[1]
            target = re.search('/(\d*)-', input_file).group()[1:-1]
            
            # output result/index.0.bin => index range from 0 to 152
            output_file='result/'+index+'.0.bin'
            

            output = np.fromfile(output_file, np.float32)
            res = np.argmax(output)
            print('Predicted:', label[res], 'Target:', target)
            total += 1
            if label[res] != target:
                error += 1
    accuracy = float(total - error) / total * 100
    print('\nClassification Accuracy: {:.2f}%\n'.format(accuracy))
