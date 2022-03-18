# coding=utf-8
'''
Created on 2016年9月27日
@author: dengdan
'''
#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#



import numpy as np

float32 = 'float32'
floatX = float32
int32 = 'int32'
uint8 = 'uint8'
string = 'str'


def cast(obj, dtype):
    if isinstance(obj, list):
        return np.asarray(obj, dtype=floatX)
    return np.cast[dtype](obj)


def int(obj):
    return cast(obj, 'int')


def double(obj):
    return cast(obj, 'double')


def is_number(obj):
    try:
        obj + 1
    except:
        return False
    return True


def is_str(s):
    return type(s) == str


def is_list(s):
    return type(s) == list


def is_tuple(s):
    return type(s) == tuple
