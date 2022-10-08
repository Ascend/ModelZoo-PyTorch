#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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


import sys

if __name__ == '__main__':
    # path settings
    txt_path = sys.argv[1]
    dir_path = sys.argv[2]

    # read preprocess_result.txt
    with open(txt_path, 'r', encoding='utf-8') as filetext: 
        elements = filetext.readlines()
        order = 1
        for element in elements:
            # print progress
            print("reprocessing turn:" + str(order) + '/50000')
            order += 1
            element_1 = element.split()
            name = element_1[0]
            element_1 = element_1[1:]
            element_2 = [[float(element_1[i]), i] for i in range(len(element_1))]
            element_2.sort(key = lambda x: x[0], reverse=True)
            # make a new txt
            with open(dir_path + '/'+ name +'_1.txt', 'a', encoding='utf-8') as f:
                # write top-5 class
                f.writelines(str(element_2[0][1]) + ' ' + str(element_2[1][1]) + ' ' + \
                             str(element_2[2][1]) + ' ' + str(element_2[3][1]) \
                            + ' ' + str(element_2[4][1]) + '\n')
            f.close()
    filetext.close()

