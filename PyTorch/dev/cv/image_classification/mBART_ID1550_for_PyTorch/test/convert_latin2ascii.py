#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
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
#
"""
pytorch-dl
Created by raj at 12:22 PM,  2/13/20
"""

import sys
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))

map = {
    '盲': 'ae',
    '枚': 'oe',
    '眉': 'ue',
    '脛': 'Ae',
    '脰': 'Oe',
    '脺': 'Ue',
    '脽': 'ss'
}

all_caps_map = {
    '脛': 'AE',
    '脰': 'OE',
    '脺': 'UE',
    '脽': 'SS'
}

with open(sys.argv[1]) as fin, open(sys.argv[1]+'.ascii.txt', 'w') as fout, open(sys.argv[1]+'.log', 'w') as logfile:
    for line in fin:
        ascii_flag = False
        original = line
        tokens = line.split()
        tokens_updated = list()

        # use step=2 to normalize a truecaser model
        step = 1
        for i in range(0, len(tokens), step):
            word = tokens[i]
            if word.isupper():
                for latin in all_caps_map:
                    if latin in word:
                        word = word.replace(latin, all_caps_map[latin])
                        ascii_flag = True
            else:
                for latin in map:
                    if latin in word:
                        word = word.replace(latin, map[latin])
                        ascii_flag = True
            tokens[i] = word

        if ascii_flag:
            logfile.write(original)
            fout.write(" ".join(tokens) + '\n')
