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
label = {0:'163', 1:'7367', 2:'332', 3:'1970', 4:'4640', 5:'8629', 6:'6848', 7:'1088', 8:'460', 
         9:'6272', 10:'7312', 11:'2136', 12:'1867', 13:'669', 14:'3526', 15:'3664', 16:'3242', 
         17:'19', 18:'32', 19:'5789', 20:'118', 21:'226', 22:'7859', 23:'3947', 24:'1898', 25:'2416', 
         26:'1737', 27:'4680'}

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
            output_file = opt.result_dir + '/' + index + '.0.bin'
            output = np.fromfile(output_file, np.float32)
            res = np.argmax(output)
            print('Predicted:', label[res], 'Target:', target)
            total += 1
            if label[res] != target:
                error += 1
    accuracy = (total - error) / total * 100
    print('\nClassification Accuracy: {:.2f}%\n'.format(accuracy))